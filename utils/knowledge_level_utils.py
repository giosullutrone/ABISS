from dataset_dataclasses.benchmark import UserKnowledgeLevel


# Consolidated Knowledge Level Information Dictionary
# Contains all information needed for each knowledge level in one place
KNOWLEDGE_LEVEL_INFO = {
    UserKnowledgeLevel.EXPERT: {
        'description': '''**Your Database Knowledge Level:** EXPERT - You have complete technical schema knowledge.
**How to use this knowledge:** You can reference specific tables, columns, and relationships when it helps clarify your intent. However, HOW you express this knowledge should match your writing style (formal vs casual, concise vs descriptive, etc.). Think: technical knowledge + your personal communication style.''',
        
        'relevant_guidelines': '''- You have complete schema knowledge - use technical precision when it helps clarify
- Reference specific tables, columns, or schema elements if it clarifies your intent
- Example: 'I mean the email_address column in contacts, not the notification_log' ''',
        
        'technical_guidelines': '''- You can reference SQL concepts with technical precision when needed
- Examples: 'ORDER BY date DESC', 'LIMIT 10', 'COUNT(*) grouped by category' ''',
        
        'style_fusion': '''**IMPORTANT - Fusing Knowledge Level with Style:**
You have technical knowledge AND a personal communication style. Express technical details in YOUR way:
- Formal + Full: Professional technical language ('the email_address column')
- Colloquial + Full: Casual technical language ('yeah, the email_address field')
- Concise + Full: Brief technical refs ('email_address column')
- Descriptive + Full: Explained technical language ('the email_address column, which stores contact information')
- Imperative + Full: Direct technical commands ('Use the email_address column')
- Interrogative + Full: Technical questions ('Are you referring to the email_address column?')''',
        
        'technical_style_fusion': '''Express technical preferences in your communication style (formal SQL terms vs casual technical language).'''
    },
    
    UserKnowledgeLevel.DOMAIN: {
        'description': '''**Your Database Knowledge Level:** DOMAIN (Natural Language) - You understand the domain but not technical details.
You can discuss entities, relationships, and domain concepts naturally, but avoid referencing specific table or column names. Express domain expertise in your natural communication style - you know WHAT you want, just not the technical implementation. Think: domain expertise + your personal communication style.''',
        
        'relevant_guidelines': '''- You understand the domain but not the technical schema details
- Use domain concepts and natural language, avoiding technical table/column names
- Example: 'I mean the contact emails, not the notification logs' ''',
        
        'technical_guidelines': '''- Express preferences in natural language, avoiding SQL syntax
- Examples: 'Show newest first', 'Give me the top 10', 'Count how many in each category' ''',
        
        'style_fusion': '''**IMPORTANT - Fusing Knowledge Level with Style:**
You have domain expertise AND a personal communication style. Express domain knowledge in YOUR way:
- Formal + NL: Professional domain language ('the contact email addresses')
- Colloquial + NL: Casual domain language ('the emails from contacts')
- Concise + NL: Brief domain refs ('contact emails')
- Descriptive + NL: Explained domain language ('the email addresses we use to contact people')
- Imperative + NL: Direct domain commands ('Use the contact emails')
- Interrogative + NL: Domain questions ('Are you asking about the contact emails?')''',
        
        'technical_style_fusion': '''Express preferences using natural domain language in your communication style (avoid SQL/technical jargon).'''
    },
    
    UserKnowledgeLevel.CASUAL: {
        'description': '''**Your Database Knowledge Level:** CASUAL - You have limited schema knowledge. Answer based on domain intuition and common sense.
You're expressing what you want as any typical user would, without technical or deep domain knowledge. Your communication style is all you have - be genuine and express intent through everyday language and reasoning. Think: common sense + your personal communication style.''',
        
        'relevant_guidelines': '''- You have limited schema knowledge - answer based on domain intuition
- Express your intent using everyday language and common sense, but still provide correct information with uncertainty
- Example: 'I mean the actual email addresses people use to contact, not those system notifications' ''',
        
        'technical_guidelines': '''- Express preferences vaguely but correctly, or with uncertainty
- Examples: 'Sorted somehow, maybe most recent?', 'Just the 10 most important ones', 'Maybe count them for how many in each category' ''',
        
        'style_fusion': '''**IMPORTANT - Fusing Knowledge Level with Style:**
You have intuition AND a personal communication style. Express common sense in YOUR way:
- Formal + None: Professional everyday language ('the email addresses utilized for contact')
- Colloquial + None: Casual everyday language ('the emails people use to reach out')
- Concise + None: Brief everyday refs ('contact emails')
- Descriptive + None: Explained everyday language ('the email addresses people give us to stay in touch')
- Imperative + None: Direct everyday commands ('Use the contact emails')
- Interrogative + None: Everyday questions ('Do you mean the emails for contacting people?')''',
        
        'technical_style_fusion': '''Express preferences using everyday language and intuition in your communication style.'''
    }
}
