"""
ATS-Compliant PDF Resume Generator.
Compiles structured TailoredResume data into clean, single-column, ATS-parseable PDF documents.
Uses ReportLab with standard typography and layout hierarchy.
"""

import io
from pathlib import Path
from typing import Optional, Union
from schemas.tailoring_schema import TailoredResume


def compile_tailored_pdf(
    resume: TailoredResume,
    output_path: Optional[Union[str, Path]] = None
) -> bytes:
    """
    Compile TailoredResume into a clean, single-column, ATS-parseable PDF.

    Args:
        resume: The validated TailoredResume model.
        output_path: Optional file path to save the generated PDF.

    Returns:
        Bytes of the compiled PDF.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            HRFlowable,
        )
    except ImportError:
        raise ImportError("ReportLab is required for PDF generation. Install with 'pip install reportlab'.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,  # 0.5 inch margins
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Typography Palette for ATS Readability
    name_style = ParagraphStyle(
        "CandidateName",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,  # Centered
        textColor=colors.HexColor("#1A202C")
    )

    contact_style = ParagraphStyle(
        "ContactInfo",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#4A5568")
    )

    section_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2D3748")
    )

    item_title_style = ParagraphStyle(
        "ItemTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1A202C")
    )

    bullet_style = ParagraphStyle(
        "BulletItem",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12,
        leftIndent=14,
        firstLineIndent=-10,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=2
    )

    story = []

    # 1. Candidate Name & Contact Header
    story.append(Paragraph(resume.candidate_name.upper(), name_style))
    story.append(Spacer(1, 3))

    contact_parts = []
    if resume.contact.location:
        contact_parts.append(resume.contact.location)
    if resume.contact.email:
        contact_parts.append(resume.contact.email)
    if resume.contact.phone:
        contact_parts.append(resume.contact.phone)
    if resume.contact.linkedin_url:
        contact_parts.append(resume.contact.linkedin_url)
    if resume.contact.github_url:
        contact_parts.append(resume.contact.github_url)

    story.append(Paragraph(" | ".join(contact_parts), contact_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=6))

    # 2. Professional Summary
    story.append(Paragraph("PROFESSIONAL SUMMARY", section_heading_style))
    story.append(Paragraph(resume.professional_summary, body_style))
    story.append(Spacer(1, 4))

    # 3. Technical Skills Matrix
    story.append(Paragraph("TECHNICAL SKILLS", section_heading_style))
    skills = resume.prioritized_skills
    if skills.languages:
        story.append(Paragraph(f"<b>Languages:</b> {', '.join(skills.languages)}", body_style))
    if skills.frameworks_libraries:
        story.append(Paragraph(f"<b>Frameworks & Libraries:</b> {', '.join(skills.frameworks_libraries)}", body_style))
    if skills.databases:
        story.append(Paragraph(f"<b>Databases:</b> {', '.join(skills.databases)}", body_style))
    if skills.cloud_devops:
        story.append(Paragraph(f"<b>Cloud & DevOps:</b> {', '.join(skills.cloud_devops)}", body_style))
    if skills.developer_tools:
        story.append(Paragraph(f"<b>Developer Tools:</b> {', '.join(skills.developer_tools)}", body_style))
    story.append(Spacer(1, 4))

    # 4. Work Experience & Internships
    if resume.tailored_experiences:
        story.append(Paragraph("WORK EXPERIENCE", section_heading_style))
        for exp in resume.tailored_experiences:
            date_loc = f"{exp.start_date or ''} - {exp.end_date or 'Present'}"
            if exp.location:
                date_loc += f" | {exp.location}"

            header_text = f"<b>{exp.role}</b> — <i>{exp.company}</i> <font color='#718096'>({date_loc})</font>"
            story.append(Paragraph(header_text, item_title_style))

            for bullet in exp.tailored_bullets:
                story.append(Paragraph(f"• {bullet.tailored_text}", bullet_style))
            story.append(Spacer(1, 3))

    # 5. Projects
    if resume.tailored_projects:
        story.append(Paragraph("KEY PROJECTS", section_heading_style))
        for proj in resume.tailored_projects:
            tech_str = f" <font color='#4A5568'>[Tech: {', '.join(proj.technologies)}]</font>" if proj.technologies else ""
            story.append(Paragraph(f"<b>{proj.title}</b>{tech_str}", item_title_style))

            for bullet in proj.tailored_bullets:
                story.append(Paragraph(f"• {bullet.tailored_text}", bullet_style))
            story.append(Spacer(1, 3))

    # 6. Education
    if resume.education:
        story.append(Paragraph("EDUCATION", section_heading_style))
        for edu in resume.education:
            year_str = f"{edu.start_year} - {edu.end_year}" if edu.start_year and edu.end_year else (edu.end_year or "")
            gpa_str = f" | CGPA: {edu.grade_or_gpa}" if edu.grade_or_gpa else ""
            edu_text = f"<b>{edu.degree}</b> — {edu.institution} <font color='#718096'>({year_str}{gpa_str})</font>"
            story.append(Paragraph(edu_text, body_style))
        story.append(Spacer(1, 4))

    # 7. Certifications
    if resume.certifications:
        story.append(Paragraph("CERTIFICATIONS", section_heading_style))
        for cert in resume.certifications:
            story.append(Paragraph(f"• {cert}", bullet_style))

    # Build PDF into memory
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    # Save to disk if path provided
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f:
            f.write(pdf_bytes)
        print(f"[PDFGenerator] Saved ATS PDF to: {out_file}")

    return pdf_bytes
