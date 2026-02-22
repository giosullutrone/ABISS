"""Tests for _compare_results_relaxed and compare_query_results in db_datasets/db_dataset.py"""
import pytest
import time
import random
from db_datasets.db_dataset import DBDataset


class FakeDBDataset:
    """Minimal stub to test _compare_results_relaxed without a real database."""

    def __init__(self):
        pass

    def compare(self, cols_gen, data_gen, cols_gt, data_gt):
        return DBDataset._compare_results_relaxed(
            self, cols_gen, data_gen, cols_gt, data_gt
        )


@pytest.fixture
def db():
    return FakeDBDataset()


# =============================================================================
# 1. Exact match
# =============================================================================
class TestExactMatch:
    def test_identical_results(self, db):
        cols = ["name", "age"]
        data = [("Alice", 30), ("Bob", 25)]
        assert db.compare(cols, data, cols, data) is True

    def test_single_row_single_col(self, db):
        assert db.compare(["id"], [(1,)], ["id"], [(1,)]) is True

    def test_single_row_multi_col(self, db):
        cols = ["a", "b", "c"]
        data = [("x", 1, 3.14)]
        assert db.compare(cols, data, cols, data) is True

    def test_many_rows_identical(self, db):
        cols = ["v"]
        data = [(i,) for i in range(100)]
        assert db.compare(cols, data, cols, data) is True


# =============================================================================
# 2. Column names are ignored (the core fix)
# =============================================================================
class TestColumnNameIgnored:
    def test_different_aliases_same_data(self, db):
        """The original bug: number_of_students vs NumberOfStudents."""
        cols_gen = ["School", "City", "Zip", "NumberOfStudents"]
        cols_gt = ["School", "City", "Zip", "number_of_students"]
        data = [("Lincoln High", "Oakland", "94601", 650),
                ("Roosevelt Middle", "Berkeley", "94703", 520)]
        assert db.compare(cols_gen, data, cols_gt, data) is True

    def test_completely_different_names(self, db):
        cols_gen = ["a", "b"]
        cols_gt = ["x", "y"]
        data = [("hello", 1), ("world", 2)]
        assert db.compare(cols_gen, data, cols_gt, data) is True

    def test_alias_casing_only_difference(self, db):
        cols_gen = ["AvgScore"]
        cols_gt = ["avg_score"]
        data = [(95.5,), (87.3,)]
        assert db.compare(cols_gen, data, cols_gt, data) is True

    def test_underscore_vs_camelcase(self, db):
        cols_gen = ["firstName", "lastName"]
        cols_gt = ["first_name", "last_name"]
        data = [("Alice", "Smith"), ("Bob", "Jones")]
        assert db.compare(cols_gen, data, cols_gt, data) is True

    def test_aggregation_aliases(self, db):
        """COUNT(*) AS count vs COUNT(*) AS total_count — common in generated SQL."""
        cols_gen = ["dept", "total_count"]
        cols_gt = ["department", "count"]
        data = [("Engineering", 42), ("Sales", 15)]
        assert db.compare(cols_gen, data, cols_gt, data) is True


# =============================================================================
# 3. Column superset / subset
# =============================================================================
class TestColumnSuperset:
    def test_one_extra_column(self, db):
        cols_gen = ["name", "age", "state"]
        data_gen = [("Alice", 30, "CA"), ("Bob", 25, "NY")]
        cols_gt = ["n", "a"]
        data_gt = [("Alice", 30), ("Bob", 25)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_many_extra_columns(self, db):
        """Generated returns 5 columns, GT only needs 2."""
        cols_gen = ["a", "b", "c", "d", "e"]
        data_gen = [(1, "x", 10, 100, "z"), (2, "y", 20, 200, "w")]
        cols_gt = ["x", "y"]
        data_gt = [(1, "x"), (2, "y")]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_generated_has_fewer_columns(self, db):
        cols_gen = ["name"]
        data_gen = [("Alice",), ("Bob",)]
        cols_gt = ["name", "age"]
        data_gt = [("Alice", 30), ("Bob", 25)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is False

    def test_exact_same_column_count(self, db):
        cols_gen = ["a", "b"]
        cols_gt = ["x", "y"]
        data = [(1, 2), (3, 4)]
        assert db.compare(cols_gen, data, cols_gt, data) is True

    def test_superset_with_extra_matching_value(self, db):
        """Extra column has values that appear in GT columns — must not confuse matching."""
        cols_gen = ["id", "val", "val_copy"]
        data_gen = [(1, "a", "a"), (2, "b", "b")]
        cols_gt = ["i", "v"]
        data_gt = [(1, "a"), (2, "b")]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True


# =============================================================================
# 4. Row ordering must match (ORDER BY / LIMIT correctness)
# =============================================================================
class TestRowOrderMatters:
    def test_same_order_matches(self, db):
        cols = ["id", "val"]
        data = [(1, "a"), (2, "b")]
        assert db.compare(cols, data, cols, data) is True

    def test_reversed_rows_fails(self, db):
        """Different row order means ORDER BY mismatch."""
        cols = ["id", "val"]
        data_gen = [(2, "b"), (1, "a")]
        data_gt = [(1, "a"), (2, "b")]
        assert db.compare(cols, data_gen, cols, data_gt) is False

    def test_shuffled_rows_fails(self, db):
        cols = ["id"]
        data_gen = [(3,), (1,), (2,)]
        data_gt = [(1,), (2,), (3,)]
        assert db.compare(cols, data_gen, cols, data_gt) is False

    def test_columns_reordered_but_rows_same_order(self, db):
        """Column order can differ but row order must match."""
        cols_gen = ["age", "name"]
        data_gen = [(30, "Alice"), (25, "Bob")]
        cols_gt = ["n", "a"]
        data_gt = [("Alice", 30), ("Bob", 25)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_columns_reordered_and_rows_reordered_fails(self, db):
        """Column order can differ but row order must still match."""
        cols_gen = ["age", "name"]
        data_gen = [(25, "Bob"), (30, "Alice")]
        cols_gt = ["n", "a"]
        data_gt = [("Alice", 30), ("Bob", 25)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is False

    def test_limit_same_order(self, db):
        """LIMIT with same ORDER BY — same rows in same order."""
        cols_gen = ["score"]
        data_gen = [(95,), (90,), (88,)]
        cols_gt = ["avg_score"]
        data_gt = [(95,), (90,), (88,)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_limit_different_order_fails(self, db):
        """LIMIT with different ORDER BY (ASC vs DESC) — different row order."""
        cols_gen = ["score"]
        data_gen = [(95,), (90,), (88,)]
        cols_gt = ["avg_score"]
        data_gt = [(88,), (90,), (95,)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is False

    def test_limit_different_rows_fails(self, db):
        """Different LIMIT selects different rows entirely."""
        data_gen = [(1,), (2,), (3,)]
        data_gt = [(1,), (2,)]
        assert db.compare(["a"], data_gen, ["b"], data_gt) is False


# =============================================================================
# 5. Column order is ignored
# =============================================================================
class TestColumnOrderIgnored:
    def test_two_columns_swapped(self, db):
        cols_gen = ["age", "name"]
        data_gen = [(30, "Alice"), (25, "Bob")]
        cols_gt = ["n", "a"]
        data_gt = [("Alice", 30), ("Bob", 25)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_three_columns_rotated(self, db):
        cols_gen = ["c", "a", "b"]
        data_gen = [(3, 1, 2), (6, 4, 5)]
        cols_gt = ["x", "y", "z"]
        data_gt = [(1, 2, 3), (4, 5, 6)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True


# =============================================================================
# 6. Data mismatches (should fail)
# =============================================================================
class TestDifferentData:
    def test_different_values_single_col(self, db):
        assert db.compare(["a"], [(1,), (2,)], ["a"], [(1,), (3,)]) is False

    def test_different_values_multi_col(self, db):
        data_gen = [("Alice", 30), ("Bob", 25)]
        data_gt = [("Alice", 30), ("Bob", 26)]  # age differs
        assert db.compare(["a", "b"], data_gen, ["a", "b"], data_gt) is False

    def test_different_row_count_more(self, db):
        assert db.compare(["a"], [(1,), (2,), (3,)], ["a"], [(1,), (2,)]) is False

    def test_different_row_count_fewer(self, db):
        assert db.compare(["a"], [(1,)], ["a"], [(1,), (2,)]) is False

    def test_completely_disjoint(self, db):
        assert db.compare(["a"], [(1,), (2,)], ["a"], [(3,), (4,)]) is False

    def test_subset_of_rows(self, db):
        """Generated returns a subset of GT rows (same count but different content)."""
        assert db.compare(["a"], [(1,), (1,)], ["a"], [(1,), (2,)]) is False


# =============================================================================
# 7. Empty results
# =============================================================================
class TestEmptyResults:
    def test_both_empty(self, db):
        assert db.compare(["a"], [], ["a"], []) is True

    def test_both_empty_different_schemas(self, db):
        assert db.compare(["a", "b"], [], ["x"], []) is True

    def test_generated_empty_gt_not(self, db):
        assert db.compare(["a"], [], ["a"], [(1,)]) is False

    def test_gt_empty_generated_not(self, db):
        assert db.compare(["a"], [(1,)], ["a"], []) is False


# =============================================================================
# 8. NULL / None values
# =============================================================================
class TestNoneValues:
    def test_null_values_match(self, db):
        data = [(1, None), (2, "ok")]
        assert db.compare(["a", "b"], data, ["a", "b"], data) is True

    def test_all_null_column(self, db):
        data = [(None,), (None,), (None,)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_null_vs_non_null(self, db):
        data_gen = [(None,)]
        data_gt = [("",)]
        assert db.compare(["a"], data_gen, ["a"], data_gt) is False

    def test_null_vs_zero(self, db):
        data_gen = [(None,)]
        data_gt = [(0,)]
        assert db.compare(["a"], data_gen, ["a"], data_gt) is False

    def test_mixed_nulls_different_positions(self, db):
        data_gen = [(1, None), (None, 2)]
        data_gt = [(1, None), (None, 2)]
        assert db.compare(["a", "b"], data_gen, ["x", "y"], data_gt) is True


# =============================================================================
# 9. Duplicate rows (set semantics)
# =============================================================================
class TestDuplicateRows:
    def test_duplicate_rows_both_sides(self, db):
        """When both have the same duplicates, set() collapses them equally."""
        data = [(1,), (1,), (2,)]
        # set() makes this {(1,), (2,)} on both sides — same row count so passes
        assert db.compare(["a"], data, ["b"], data) is True

    def test_different_duplicate_counts(self, db):
        """Different duplicate distributions are correctly rejected.
        gen=(1,1,2) vs gt=(1,2,2) — same row count but different value frequencies."""
        data_gen = [(1,), (1,), (2,)]
        data_gt = [(1,), (2,), (2,)]
        assert db.compare(["a"], data_gen, ["a"], data_gt) is False


# =============================================================================
# 10. Type handling
# =============================================================================
class TestTypeHandling:
    def test_int_values(self, db):
        data = [(1,), (2,), (3,)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_float_values(self, db):
        data = [(1.5,), (2.7,), (3.14,)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_string_values(self, db):
        data = [("hello",), ("world",)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_mixed_types_across_columns(self, db):
        """Different columns can have different types."""
        data = [(1, "Alice", 3.14), (2, "Bob", 2.72)]
        assert db.compare(["a", "b", "c"], data, ["x", "y", "z"], data) is True

    def test_int_vs_float_mismatch(self, db):
        """SQLite may return 1 vs 1.0 — these are not equal in Python."""
        data_gen = [(1,)]
        data_gt = [(1.0,)]
        # In Python, 1 == 1.0 is True and hash(1) == hash(1.0), so they match in sets
        assert db.compare(["a"], data_gen, ["a"], data_gt) is True

    def test_bool_values(self, db):
        """SQLite doesn't have native bool, but Python might pass them."""
        data = [(True,), (False,)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_bytes_values(self, db):
        """SQLite BLOB columns return bytes."""
        data = [(b"abc",), (b"def",)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_empty_string_vs_null(self, db):
        assert db.compare(["a"], [("",)], ["a"], [(None,)]) is False


# =============================================================================
# 11. Ambiguous column matching (backtracking)
# =============================================================================
class TestAmbiguousColumns:
    def test_two_columns_same_multiset_correct_pairing(self, db):
        """Columns share the same value multisets but correct row pairing exists."""
        data_gen = [(1, 2), (2, 1)]
        data_gt = [(1, 2), (2, 1)]
        assert db.compare(["x", "y"], data_gen, ["a", "b"], data_gt) is True

    def test_two_columns_same_multiset_wrong_pairing(self, db):
        """Same column multisets but no valid row pairing after projection."""
        data_gen = [(1, 2), (2, 3)]
        data_gt = [(2, 2), (1, 3)]
        assert db.compare(["x", "y"], data_gen, ["a", "b"], data_gt) is False

    def test_identical_columns_any_assignment_works(self, db):
        data = [(1, 1), (2, 2)]
        assert db.compare(["x", "y"], data, ["a", "b"], data) is True

    def test_three_columns_two_ambiguous(self, db):
        """Three columns where two have matching multisets, requiring backtracking."""
        # col0 and col1 both have {10, 20} but col2 is unique {100, 200}
        data_gen = [(10, 20, 100), (20, 10, 200)]
        data_gt = [(10, 20, 100), (20, 10, 200)]
        assert db.compare(["a", "b", "c"], data_gen, ["x", "y", "z"], data_gt) is True

    def test_backtracking_first_candidate_wrong(self, db):
        """First candidate mapping fails, backtracking finds the correct one."""
        # col0={1,2}, col1={1,2} — same multisets
        # Identity mapping: projected=[(1,2),(2,1)] vs gt=[(2,1),(1,2)] — fails (order matters)
        # Swap mapping: projected=[(2,1),(1,2)] vs gt=[(2,1),(1,2)] — passes
        cols_gen = ["a", "b"]
        data_gen = [(1, 2), (2, 1)]
        cols_gt = ["x", "y"]
        data_gt = [(2, 1), (1, 2)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_no_candidate_for_one_column(self, db):
        """One GT column has no matching generated column — early exit."""
        data_gen = [(1, "a"), (2, "b")]
        data_gt = [(1, "c"), (2, "d")]  # second column values don't match anything
        assert db.compare(["x", "y"], data_gen, ["a", "b"], data_gt) is False

    def test_all_columns_constant_same_value(self, db):
        """All columns contain the same constant — any mapping works."""
        data = [(5, 5, 5)] * 3
        assert db.compare(["a", "b", "c"], data, ["x", "y", "z"], data) is True

    def test_all_columns_constant_different_values(self, db):
        """Each column is constant but with a different value."""
        data_gen = [(1, 2, 3)] * 4
        data_gt = [(3, 1, 2)] * 4  # permutation of columns
        assert db.compare(["a", "b", "c"], data_gen, ["x", "y", "z"], data_gt) is True


# =============================================================================
# 12. Edge cases
# =============================================================================
class TestEdgeCases:
    def test_zero_columns_both_sides(self, db):
        """Zero columns — degenerate but should handle gracefully."""
        # Both have 2 rows of empty tuples — row count matches, 0 GT cols, trivially true
        assert db.compare([], [(), ()], [], [(), ()]) is True

    def test_zero_gt_columns_with_gen_columns(self, db):
        """GT has no columns to match — trivially true if row counts match."""
        assert db.compare(["a"], [(1,), (2,)], [], [(), ()]) is True

    def test_single_value_match(self, db):
        assert db.compare(["a"], [(42,)], ["b"], [(42,)]) is True

    def test_single_value_mismatch(self, db):
        assert db.compare(["a"], [(42,)], ["b"], [(43,)]) is False

    def test_large_string_values(self, db):
        s1 = "a" * 10000
        s2 = "b" * 10000
        data = [(s1,), (s2,)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_special_characters_in_values(self, db):
        data = [("O'Brien",), ("Smith\"Jr",), ("New\nLine",)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_unicode_values(self, db):
        data = [("\u00e9\u00e8\u00ea",), ("\u00fc\u00f6\u00e4",)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_negative_numbers(self, db):
        data = [(-1,), (-2,), (0,)]
        assert db.compare(["a"], data, ["b"], data) is True

    def test_very_large_numbers(self, db):
        data = [(10**18,), (-(10**18),)]
        assert db.compare(["a"], data, ["b"], data) is True


# =============================================================================
# 13. Real-world-like scenarios
# =============================================================================
class TestRealWorldScenarios:
    def test_school_enrollment_original_bug(self, db):
        """The exact scenario that triggered the fix."""
        cols_gen = ["School", "City", "Zip", "State", "NumberOfStudents"]
        data_gen = [
            ("Lincoln High", "Oakland", "94601", "CA", 650),
            ("Roosevelt Middle", "Berkeley", "94703", "CA", 520),
        ]
        cols_gt = ["School", "City", "Zip", "number_of_students"]
        data_gt = [
            ("Lincoln High", "Oakland", "94601", 650),
            ("Roosevelt Middle", "Berkeley", "94703", 520),
        ]
        # GT is a column subset, names differ, extra State column — should match
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_aggregation_with_different_alias(self, db):
        """COUNT/SUM with different aliases."""
        cols_gen = ["dept", "employee_count"]
        data_gen = [("Engineering", 42), ("Sales", 15), ("HR", 8)]
        cols_gt = ["department_name", "cnt"]
        data_gt = [("Engineering", 42), ("Sales", 15), ("HR", 8)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_join_result_same_order(self, db):
        """JOIN results with same row ordering and different column names."""
        cols_gen = ["id", "name", "dept"]
        data_gen = [(1, "Alice", "Eng"), (2, "Bob", "Sales"), (3, "Charlie", "HR")]
        cols_gt = ["eid", "ename", "department"]
        data_gt = [(1, "Alice", "Eng"), (2, "Bob", "Sales"), (3, "Charlie", "HR")]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is True

    def test_join_result_different_order_fails(self, db):
        """JOIN results in different row order — fails because ORDER BY must match."""
        cols_gen = ["id", "name", "dept"]
        data_gen = [(3, "Charlie", "HR"), (1, "Alice", "Eng"), (2, "Bob", "Sales")]
        cols_gt = ["eid", "ename", "department"]
        data_gt = [(1, "Alice", "Eng"), (2, "Bob", "Sales"), (3, "Charlie", "HR")]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is False

    def test_order_by_desc_vs_asc_fails(self, db):
        """ORDER BY DESC vs ASC produces different row ordering."""
        cols_gen = ["score"]
        data_gen = [(95,), (90,), (88,), (85,), (80,)]
        cols_gt = ["avg_score"]
        data_gt = [(80,), (85,), (88,), (90,), (95,)]
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is False

    def test_inner_join_vs_left_join_different_results(self, db):
        """Different join types produce different row sets — should fail."""
        cols_gen = ["id", "name"]
        data_gen = [(1, "Alice"), (2, "Bob")]  # INNER JOIN: 2 rows
        cols_gt = ["id", "name"]
        data_gt = [(1, "Alice"), (2, "Bob"), (3, None)]  # LEFT JOIN: 3 rows
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is False

    def test_group_by_vs_no_group_by(self, db):
        """Collective vs distributive scope interpretation (different structure)."""
        cols_gen = ["total"]
        data_gen = [(1000,)]  # single grand total
        cols_gt = ["dept", "total"]
        data_gt = [("Eng", 600), ("Sales", 400)]  # per-department
        assert db.compare(cols_gen, data_gen, cols_gt, data_gt) is False


# =============================================================================
# 14. Symmetry and commutativity
# =============================================================================
class TestSymmetry:
    def test_superset_not_symmetric(self, db):
        """Superset is allowed for gen but not for gt — check both directions."""
        cols_wide = ["a", "b", "c"]
        data_wide = [(1, 2, 3), (4, 5, 6)]
        cols_narrow = ["x", "y"]
        data_narrow = [(1, 2), (4, 5)]
        # gen=wide, gt=narrow — should pass (superset allowed)
        assert db.compare(cols_wide, data_wide, cols_narrow, data_narrow) is True
        # gen=narrow, gt=wide — should fail (subset not allowed)
        assert db.compare(cols_narrow, data_narrow, cols_wide, data_wide) is False


# =============================================================================
# 15. Performance
# =============================================================================
class TestPerformance:
    def test_large_result_set(self, db):
        """5000 rows x 6 columns with unique distributions."""
        random.seed(42)
        n_rows = 5000
        n_cols = 6
        cols_gen = [f"col_{i}" for i in range(n_cols)]
        cols_gt = [f"gt_col_{i}" for i in range(n_cols)]
        data = [tuple(random.randint(0, 100000) for _ in range(n_cols)) for _ in range(n_rows)]

        start = time.time()
        result = db.compare(cols_gen, data, cols_gt, data)
        elapsed = time.time() - start

        assert result is True
        assert elapsed < 2.0, f"Took {elapsed:.2f}s — too slow for {n_rows} rows x {n_cols} cols"

    def test_many_columns_unique_distributions(self, db):
        """15 columns each with unique distribution — backtracking has 1 candidate each."""
        random.seed(123)
        n_rows = 100
        n_cols = 15
        cols_gen = [f"g{i}" for i in range(n_cols)]
        cols_gt = [f"t{i}" for i in range(n_cols)]
        data = [tuple(i * 1000 + j for j in range(n_cols)) for i in range(n_rows)]

        start = time.time()
        result = db.compare(cols_gen, data, cols_gt, data)
        elapsed = time.time() - start

        assert result is True
        assert elapsed < 2.0, f"Took {elapsed:.2f}s — too slow for {n_rows} rows x {n_cols} cols"

    def test_mismatch_early_exit_fast(self, db):
        """Row count mismatch should return immediately."""
        n_rows = 100000
        data_gen = [(i,) for i in range(n_rows)]
        data_gt = [(i,) for i in range(n_rows + 1)]

        start = time.time()
        result = db.compare(["a"], data_gen, ["b"], data_gt)
        elapsed = time.time() - start

        assert result is False
        assert elapsed < 0.1, f"Row count mismatch should be instant, took {elapsed:.2f}s"

    def test_no_candidate_early_exit_fast(self, db):
        """When a GT column has no matching gen column, should exit quickly."""
        n_rows = 1000
        data_gen = [(i, i * 10) for i in range(n_rows)]
        data_gt = [(i, i * 99) for i in range(n_rows)]  # second col values don't match

        start = time.time()
        result = db.compare(["a", "b"], data_gen, ["x", "y"], data_gt)
        elapsed = time.time() - start

        assert result is False
        assert elapsed < 1.0, f"No-candidate case should be fast, took {elapsed:.2f}s"
