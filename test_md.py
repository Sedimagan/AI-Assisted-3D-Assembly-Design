import re

def _markdown_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = text.splitlines()
    html_parts = []
    
    in_list = False
    para_lines = []
    
    def flush_para():
        if para_lines:
            para_text = " ".join(para_lines).strip()
            if para_text:
                para_text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#ffffff;font-weight:700;">\1</strong>', para_text)
                para_text = re.sub(r'\*(.*?)\*', r'<em style="color:#cbd5e1;font-style:italic;">\1</em>', para_text)
                html_parts.append(f'<p style="font-size:0.88rem;color:#e2e8f0;line-height:1.7;margin:0.4rem 0 0.8rem;">{para_text}</p>')
            para_lines.clear()

    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            flush_para()
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            continue
            
        m_h = re.match(r'^(#{1,6})\s+(.*?)$', stripped)
        if m_h:
            flush_para()
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            level = len(m_h.group(1))
            h_text = m_h.group(2)
            h_text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#ffffff;font-weight:700;">\1</strong>', h_text)
            h_text = re.sub(r'\*(.*?)\*', r'<em style="color:#cbd5e1;font-style:italic;">\1</em>', h_text)
            
            if level == 1:
                html_parts.append(f'<h2 style="color:#ffffff;font-size:1.1rem;font-weight:700;margin:1.2rem 0 0.6rem;">{h_text}</h2>')
            elif level == 2:
                html_parts.append(f'<h3 style="color:#ffffff;font-size:1.0rem;font-weight:700;margin:1.0rem 0 0.5rem;">{h_text}</h3>')
            else:
                html_parts.append(f'<h4 style="color:#ffffff;font-size:0.92rem;font-weight:700;margin:0.8rem 0 0.4rem;">{h_text}</h4>')
            continue
            
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_para()
            content = stripped[2:]
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#ffffff;font-weight:700;">\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em style="color:#cbd5e1;font-style:italic;">\1</em>', content)
            if not in_list:
                html_parts.append('<ul style="margin:0.4rem 0;padding-left:1.2rem;color:#f1f5f9;list-style-type:disc;">')
                in_list = True
            html_parts.append(f'<li style="margin-bottom:0.25rem;font-size:0.88rem;line-height:1.6;">{content}</li>')
            continue
            
        if in_list:
            html_parts.append('</ul>')
            in_list = False
        para_lines.append(stripped)
        
    flush_para()
    if in_list:
        html_parts.append('</ul>')
        
    return "\n".join(html_parts)

text = """### Here's an interpretation of the GNN predictions:

Mechanical Engineering Interpretation of Predictions:

**Missing Link Predictions (Nodes 26, 27, 28)**: The GNN model predicts a medium confidence (0.507) that nodes 26, 27, and 28 should be connected to each other.
- **Missing Connections**: There are physical connections between these parts.
- *Functional Grouping*: These three parts are likely designed to work together.
"""

print(_markdown_to_html(text))
