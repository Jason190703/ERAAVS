import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import numpy as np
import base64
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_pdf_report(requirements_df, total_count, met_count, partial_count, unmet_count):
    """
    Generate a PDF report summarizing the requirements analysis
    
    Args:
        requirements_df: DataFrame containing requirement analysis
        total_count: Total number of requirements
        met_count: Number of met requirements
        partial_count: Number of partially met requirements
        unmet_count: Number of unmet requirements
    
    Returns:
        BytesIO: PDF file buffer
    """
    # Create a PDF buffer
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles with unique names to avoid conflicts - more professional
    heading1_style = ParagraphStyle(
        name='CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.navy  # Professional dark blue color
    )
    
    heading2_style = ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=10,
        textColor=colors.blue,  # Blue for subheadings
        borderWidth=0,
        borderColor=colors.lightgrey,
        borderPadding=5
    )
    
    heading3_style = ParagraphStyle(
        name='CustomHeading3',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.darkblue,  # Dark blue for section headers
        fontName='Helvetica-Bold'
    )
    
    normal_justified_style = ParagraphStyle(
        name='CustomNormal_Justified',
        parent=styles['Normal'],
        alignment=4,  # Justified alignment
        fontSize=10,
        leading=14  # Line spacing for better readability
    )
    
    # Additional professional styles
    bullet_style = ParagraphStyle(
        name='BulletStyle',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        bulletIndent=10,
        bulletFontName='Symbol',
        bulletFontSize=8
    )
    
    evidence_style = ParagraphStyle(
        name='EvidenceStyle',
        parent=styles['Italic'],
        fontSize=9,
        leftIndent=30,
        textColor=colors.darkgrey
    )
    
    # Initialize content for the PDF
    content = []
    
    # Add report title
    title = Paragraph("Requirements Analysis and Verification Report", heading1_style)
    content.append(title)
    content.append(Spacer(1, 0.25*inch))
    
    # Add timestamp
    timestamp = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    content.append(timestamp)
    content.append(Spacer(1, 0.25*inch))
    
    # Add requirement analysis summary header with checkmark
    content.append(Paragraph("✅ Requirement Analysis Summary", heading2_style))
    
    # Add summary statistics
    stats_data = [
        ["Requirement Status", "Count", "Percentage"],
        ["Met", str(met_count), f"{met_count/total_count*100:.1f}%"],
        ["Partially Met", str(partial_count), f"{partial_count/total_count*100:.1f}%"],
        ["Unmet", str(unmet_count), f"{unmet_count/total_count*100:.1f}%"],
        ["Total", str(total_count), "100%"]
    ]
    
    # Create table for summary statistics
    stats_table = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    content.append(stats_table)
    content.append(Spacer(1, 0.25*inch))
    
    # Create pie chart
    try:
        fig, ax = plt.subplots(figsize=(5, 3))
        status_values = [met_count, partial_count, unmet_count]
        status_labels = ['Met', 'Partially Met', 'Unmet']
        colors_pie = ['#4CAF50', '#FFC107', '#F44336']
        
        ax.pie(
            status_values, 
            labels=status_labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_pie
        )
        ax.axis('equal')
        
        # Save the chart to a buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        
        # Add the chart to the PDF
        chart_image = Image(img_buffer, width=4*inch, height=3*inch)
        content.append(chart_image)
        content.append(Spacer(1, 0.25*inch))
        
    except Exception as e:
        logger.error(f"Failed to create chart: {str(e)}")
        content.append(Paragraph("Chart generation failed", styles['Normal']))
    
    # Add individual requirement analysis in the requested format
    content.append(Spacer(1, 0.1*inch))
    
    # Add each requirement with formatted details
    for i, (_, row) in enumerate(requirements_df.iterrows()):
        req_text = row['requirement']
        status = row['status']
        confidence = row.get('confidence', 0)
        
        # Choose icon based on status
        status_icon = ""
        if status == "Met":
            status_icon = "✅"
        elif status == "Partially Met":
            status_icon = "⚠️"
        else:
            status_icon = "❌"
        
        # Create styled requirement box with border and background
        req_box_style = ParagraphStyle(
            name=f'ReqBox{i}',
            parent=heading3_style,
            backColor=colors.whitesmoke,
            borderColor=colors.lightgrey,
            borderWidth=1,
            borderPadding=8,
            borderRadius=5
        )
        
        # Add requirement title with number - with professional styling
        content.append(Paragraph(
            f'<font color="#1E4F95">Requirement {i+1}:</font> {req_text}', 
            req_box_style
        ))
        
        # Choose status color based on status
        status_color = colors.green
        if status == "Partially Met":
            status_color = colors.orange
        elif status == "Unmet":
            status_color = colors.red
            
        # Create custom style for this status
        status_style = ParagraphStyle(
            name=f'StatusStyle{i}',
            parent=normal_justified_style,
            textColor=status_color,
            leftIndent=20,
            fontName='Helvetica-Bold'
        )
        
        # Add status line with icon and confidence
        status_text = Paragraph(
            f"Status: {status_icon} {status} (Confidence: {confidence:.2f})",
            status_style
        )
        content.append(status_text)
        
        # Add remarks/evidence with more professional styling
        remarks_style = ParagraphStyle(
            name=f'RemarksStyle{i}',
            parent=evidence_style,
            leftIndent=20,
            firstLineIndent=0,
            bulletIndent=0,
            spaceBefore=6,
            spaceAfter=8
        )
        
        if 'evidence' in row and row['evidence'] and row['evidence'] != "No direct evidence found.":
            remarks = Paragraph(f"<b>Remarks:</b> {row['evidence']}", remarks_style)
        elif status == "Met":
            remarks = Paragraph("<b>Remarks:</b> Requirement fully implemented in the product documentation.", remarks_style)
        elif status == "Partially Met":
            remarks = Paragraph("<b>Remarks:</b> Requirement partially addressed; additional clarification needed.", remarks_style)
        else:
            remarks = Paragraph("<b>Remarks:</b> Requirement not found in product documentation.", remarks_style)
        
        content.append(remarks)
        content.append(Spacer(1, 0.15*inch))
    
    content.append(Spacer(1, 0.1*inch))
    
    # Add summary section with emoji
    content.append(Paragraph("📊 Summary", heading2_style))
    
    # Group requirements by status
    met_reqs = requirements_df[requirements_df['status'] == 'Met']['requirement'].tolist()
    partial_reqs = requirements_df[requirements_df['status'] == 'Partially Met']['requirement'].tolist()
    unmet_reqs = requirements_df[requirements_df['status'] == 'Unmet']['requirement'].tolist()
    
    # Create a more professional summary table with counts and icons
    summary_data = [
        ["Metric", "Count", "Percentage"],
        ["Total Requirements", str(total_count), "100%"],
        ["Requirements Met ✅", str(met_count), f"{(met_count/total_count*100):.1f}%" if total_count > 0 else "0%"],
        ["Requirements Partially Met ⚠️", str(partial_count), f"{(partial_count/total_count*100):.1f}%" if total_count > 0 else "0%"],
        ["Requirements Not Met ❌", str(unmet_count), f"{(unmet_count/total_count*100):.1f}%" if total_count > 0 else "0%"]
    ]
    
    # Calculate completion rate
    completion_rate = ((met_count + (0.5 * partial_count)) / total_count * 100) if total_count > 0 else 0
    summary_data.append(["Overall Completion Rate", f"{completion_rate:.1f}%", ""])
    
    # Define professional colors - using standard colors instead of custom colors
    header_bg = colors.darkblue
    met_color = colors.lightgreen
    partial_color = colors.lightyellow
    unmet_color = colors.lightpink
    completion_bg = colors.lightblue
    
    # Calculate table style with alternating row colors and professional formatting
    table_style = TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # Data rows with status-based colors
        ('BACKGROUND', (0, 1), (-1, 1), colors.whitesmoke),  # Total row
        ('BACKGROUND', (0, 2), (-1, 2), met_color),  # Met row
        ('BACKGROUND', (0, 3), (-1, 3), partial_color),  # Partially Met row
        ('BACKGROUND', (0, 4), (-1, 4), unmet_color),  # Unmet row
        ('BACKGROUND', (0, 5), (-1, 5), completion_bg),  # Completion rate row
        
        # Make percentage column right-aligned
        ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
        
        # Text formatting
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),  # Bold first column
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),  # Bold last row
        
        # Borders and grid
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),  # Thicker line below header
        ('LINEABOVE', (0, -1), (-1, -1), 1, colors.black),  # Line above final row
    ])
    
    # Create the table and apply style
    summary_table = Table(summary_data, colWidths=[250, 100, 100])
    summary_table.setStyle(table_style)
    content.append(summary_table)
    content.append(Spacer(1, 0.25*inch))
    
    # Limit to max 10 requirements for the PDF report to keep it concise
    max_details = min(10, len(requirements_df))
    
    for i in range(max_details):
        row = requirements_df.iloc[i]
        
        req_title = Paragraph(f"Requirement {i+1}: {row['requirement']}", heading2_style)
        content.append(req_title)
        
        req_status = Paragraph(f"Status: {row['status']}", normal_justified_style)
        content.append(req_status)
        
        req_confidence = Paragraph(f"Confidence Score: {row['confidence']:.2f}", normal_justified_style)
        content.append(req_confidence)
        
        # Indicate if this analysis was enhanced with AI
        if 'ai_enhanced' in row and row['ai_enhanced']:
            ai_enhanced_text = Paragraph("✓ AI Enhanced Analysis", normal_justified_style)
            content.append(ai_enhanced_text)
            
            # Include the AI model used if available
            if 'source' in row:
                source = row['source']
                if source == 'ai_openai':
                    source_text = "Analysis powered by OpenAI"
                elif source == 'ai_anthropic':
                    source_text = "Analysis powered by Anthropic Claude"
                else:
                    source_text = f"Analysis source: {source}"
                    
                source_paragraph = Paragraph(source_text, normal_justified_style)
                content.append(source_paragraph)
        
        # Include key terms if available
        if 'key_terms' in row:
            key_terms_text = Paragraph(f"Key Terms: {row['key_terms']}", normal_justified_style)
            content.append(key_terms_text)
        
        # Include found terms
        if 'found_terms' in row and row['found_terms'] != 'None':
            found_terms_text = Paragraph(f"Found Terms: {row['found_terms']}", normal_justified_style)
            content.append(found_terms_text)
        
        # Include missing terms
        if 'missing_terms' in row and row['missing_terms'] != 'None':
            missing_terms_text = Paragraph(f"Missing Terms: {row['missing_terms']}", normal_justified_style)
            content.append(missing_terms_text)
            
        # Include AI explanation if available
        if 'explanation' in row and row['explanation']:
            explanation_text = Paragraph(f"Analysis Explanation: {row['explanation']}", normal_justified_style)
            content.append(explanation_text)
        
        if 'evidence' in row and row['evidence']:
            # Format evidence text with line breaks for better readability
            evidence_text = row['evidence'].replace('\n', '<br/>')
            req_evidence = Paragraph(f"Evidence: {evidence_text}", normal_justified_style)
            content.append(req_evidence)
        
        if row['status'] == 'Unmet' or row['status'] == 'Partially Met':
            req_action = Paragraph(
                "Suggested Action: Review product documentation and ensure this requirement is properly addressed.",
                normal_justified_style
            )
            content.append(req_action)
        
        content.append(Spacer(1, 0.25*inch))
    
    # Add note if some requirements were omitted
    if len(requirements_df) > max_details:
        note = Paragraph(
            f"Note: {len(requirements_df) - max_details} additional requirements were analyzed but not included in this detailed section to keep the report concise.",
            normal_justified_style
        )
        content.append(note)
        content.append(Spacer(1, 0.25*inch))
    
    # Add conclusion in the requested format
    content.append(Paragraph("Conclusion:", heading2_style))
    
    # Determine completion word based on analysis
    completion_word = ""
    completion_rate = ((met_count + (0.5 * partial_count)) / total_count * 100) if total_count > 0 else 0
    
    if completion_rate == 100:
        completion_word = "fully"
    elif completion_rate >= 75:
        completion_word = "mostly"
    elif completion_rate >= 50:
        completion_word = "partially"
    else:
        completion_word = "minimally"
    
    # Create formatted conclusion
    conclusion_text = f"""
    Based on the analysis, the product documentation {completion_word} satisfies the client's requirements at {completion_rate:.1f}% completion.
    """
    
    if unmet_count > 0:
        conclusion_text += f" There are {unmet_count} requirements that need to be addressed in the product documentation."
    elif partial_count > 0:
        conclusion_text += f" There are {partial_count} requirements that need further clarification or enhancement."
    else:
        conclusion_text += " All requirements have been successfully implemented."
    
    content.append(Paragraph(conclusion_text, normal_justified_style))
    
    # Add footer
    footer_text = "Generated by ERAAVS - End-to-end Requirement Analysis and Verification System"
    footer = Paragraph(footer_text, styles['Normal'])
    content.append(Spacer(1, 0.5*inch))
    content.append(footer)
    
    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    
    return buffer
