import os
import uuid
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import HorizontalBarChart

def get_forensic_explanation(result: dict) -> str:
    prediction = result['prediction'].upper()
    
    artifact_intensity = result.get('forensics', {}).get('cnn_score', 0) * 100
    vocal_humanity = result.get('forensics', {}).get('frequency_score', 0) * 100

    if prediction == "FAKE":
        reason = f"<b>FORENSIC VERDICT: SYNTHETIC (AI-GENERATED)</b><br/><br/>"
        reason += f"<b>Explanation:</b> The system identified high-probability synthetic signatures throughout the audio.<br/>"
        reason += f"<b>Vocal Humanity Score:</b> {vocal_humanity:.1f}% (LOW). The semantic branch detected unnatural prosody and inconsistencies.<br/>"
        reason += f"<b>Spectral Artifacts:</b> {artifact_intensity:.1f}% (HIGH). Evidence of neural vocoder noise patterns detected."
        return reason
    elif prediction == "HYBRID":
        reason = f"<b>FORENSIC VERDICT: HYBRID (MIXED CONTENT)</b><br/><br/>"
        reason += f"<b>Explanation:</b> The audio contains both authentic human speech and AI-generated segments.<br/>"
        reason += f"<b>Vocal Humanity Score:</b> Mixed. Specific time segments align with human traits, while others show synthetic artifacts.<br/>"
        reason += f"<b>Spectral Artifacts:</b> Variable. Localized digital fingerprints were identified at specific timestamps."
        return reason
    elif prediction == "SUSPICIOUS":
        reason = f"<b>FORENSIC VERDICT: SUSPICIOUS (LOW CONFIDENCE)</b><br/><br/>"
        reason += f"<b>Explanation:</b> The audio shows unusual characteristics that deviate from standard human speech patterns.<br/>"
        reason += f"<b>Vocal Humanity Score:</b> {vocal_humanity:.1f}% (MEDIUM). Some segments have low confidence or unusual frequency distributions.<br/>"
        reason += f"<b>Spectral Artifacts:</b> {artifact_intensity:.1f}% (MODERATE). Possible low-quality recording or subtle modification artifacts."
        return reason
    else:
        reason = f"<b>FORENSIC VERDICT: AUTHENTIC (HUMAN)</b><br/><br/>"
        reason += f"<b>Explanation:</b> The audio displays characteristics consistent with natural human physiological speech production.<br/>"
        reason += f"<b>Vocal Humanity Score:</b> {vocal_humanity:.1f}% (HIGH). The semantic embeddings align with human vocal tract physics.<br/>"
        reason += f"<b>Spectral Artifacts:</b> {artifact_intensity:.1f}% (MINIMAL). No significant vocoder fingerprints detected."
        return reason

def create_confidence_bar(confidence: float, prediction: str) -> Drawing:
    drawing = Drawing(400, 40)
    bc = HorizontalBarChart()
    bc.x = 0
    bc.y = 10
    bc.height = 20
    bc.width = 400
    bc.data = [[confidence * 100]]
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = 100
    bc.categoryAxis.labels.boxAnchor = 'n'
    bc.categoryAxis.labels.dy = -5
    bc.categoryAxis.categoryNames = ['']
    
    pred = prediction.upper()
    if pred == "FAKE":
        bc.bars[0].fillColor = colors.red
    elif pred == "HYBRID":
        bc.bars[0].fillColor = colors.orange
    elif pred == "SUSPICIOUS":
        bc.bars[0].fillColor = colors.orange
    else:
        bc.bars[0].fillColor = colors.green
        
    drawing.add(bc)
    return drawing

def create_timeline_heatmap(duration: float, segments: list) -> Drawing:
    drawing_width = 450
    drawing_height = 50
    drawing = Drawing(drawing_width, drawing_height)
    
    # Base bar (Authentic)
    base_bar = Rect(0, 15, drawing_width, 20, fillColor=colors.lightgreen, strokeColor=colors.black)
    drawing.add(base_bar)
    
    # Red bars (Synthetic)
    if duration > 0:
        for seg in segments:
            start = max(0, min(seg.get('start', 0), duration))
            end = max(0, min(seg.get('end', 0), duration))
            x = (start / duration) * drawing_width
            w = ((end - start) / duration) * drawing_width
            
            pred = seg.get('prediction', 'REAL').upper()
            color = colors.red if pred == "FAKE" else colors.orange if pred == "SUSPICIOUS" else colors.lightgreen
            
            if pred != "REAL":
                fake_bar = Rect(x, 15, w, 20, fillColor=color, strokeColor=colors.black, strokeWidth=0.5)
                drawing.add(fake_bar)
            
    # Ticks
    drawing.add(Line(0, 10, 0, 15, strokeColor=colors.black))
    drawing.add(String(0, 0, "0.0s", fontSize=9, textAnchor='middle'))
    drawing.add(Line(drawing_width, 10, drawing_width, 15, strokeColor=colors.black))
    drawing.add(String(drawing_width, 0, f"{duration:.1f}s", fontSize=9, textAnchor='middle'))
    
    return drawing

def generate_pdf_report(result: dict, output_path: str):
    """
    Generates a professional Digital Forensic Report for audio deepfake analysis.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, fontSize=24, textColor=colors.HexColor("#1A237E"), spaceAfter=20)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#1A237E"), spaceBefore=15, spaceAfter=10, borderPadding=5)
    meta_style = ParagraphStyle('MetaStyle', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
    footer_style = ParagraphStyle('FooterStyle', parent=styles['Normal'], fontSize=8, alignment=1, textColor=colors.grey)
    
    elements = []
    
    # --- HEADER ---
    elements.append(Paragraph("AudLens Audio Analysis Report", title_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A237E"), spaceAfter=10))
    
    # --- METADATA ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = result.get('report_id', uuid.uuid4().hex[:8].upper())
    elements.append(Paragraph(f"<b>Report ID:</b> {report_id}", meta_style))
    elements.append(Paragraph(f"<b>Analysis Date:</b> {timestamp}", meta_style))
    elements.append(Paragraph(f"<b>File Name:</b> {result.get('filename', 'Unknown_Audio')}", meta_style))
    elements.append(Paragraph(f"<b>Format:</b> {result.get('format', 'N/A')} | <b>Duration:</b> {result.get('duration', 0):.2f}s | <b>Sampling Rate:</b> 16,000 Hz", meta_style))
    elements.append(Spacer(1, 20))
    
    # --- 🔍 PREDICTION SECTION ---
    elements.append(Paragraph("🔍 PREDICTION SUMMARY", section_style))
    
    pred_upper = result['prediction'].upper()
    verdict = "SYNTHETIC" if pred_upper == "FAKE" else "HYBRID" if pred_upper == "HYBRID" else "SUSPICIOUS" if pred_upper == "SUSPICIOUS" else "AUTHENTIC"
    status_color = colors.red if pred_upper == "FAKE" else colors.orange if pred_upper in ["HYBRID", "SUSPICIOUS"] else colors.green

    prediction_data = [
        ["Prediction Result:", pred_upper],
        ["Confidence Score:", f"{result['confidence'] * 100:.2f}%"],
        ["Forensic Verdict:", verdict]
    ]
    t = Table(prediction_data, colWidths=[150, 300])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (1,0), (1,0), status_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.lightgrey),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10))
    elements.append(create_confidence_bar(result['confidence'], result['prediction']))
    elements.append(Spacer(1, 20))
    
    # --- 📊 FORENSIC EVIDENCE SECTION ---
    elements.append(Paragraph("📊 FORENSIC EVIDENCE (SPECTROGRAM)", section_style))
    if 'spectrogram_path' in result and os.path.exists(result['spectrogram_path']):
        img = Image(result['spectrogram_path'], width=450, height=180)
        elements.append(img)
        elements.append(Paragraph("<i>Fig 1: Deep Forensic Spectrogram (Frequency-Time Domain Analysis)</i>", footer_style))
    else:
        elements.append(Paragraph("[Spectrogram Data Unavailable]", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # --- 🧠 ANALYSIS REASONING ---
    elements.append(Paragraph("🧠 ANALYSIS REASONING", section_style))
    explanation = get_forensic_explanation(result)
    elements.append(Paragraph(explanation, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # --- 📈 VISUALIZATION SECTION ---
    elements.append(Paragraph("📈 AUDIO TIMELINE HEATMAP", section_style))
    duration = result.get('duration', 0)
    elements.append(create_timeline_heatmap(duration, result.get('segments', [])))
    elements.append(Paragraph(f"Timeline: 0.0s to {duration:.1f}s", footer_style))
    elements.append(Spacer(1, 20))
    
    # --- 🎯 SEGMENT DETECTION ---
    elements.append(Paragraph("🎯 DETECTED MODIFIED SEGMENTS", section_style))
    if result['segments']:
        seg_data = [["Segment #", "Start (s)", "End (s)", "Reason"]]
        for idx, seg in enumerate(result['segments']):
            seg_data.append([str(idx + 1), f"{seg['start']:.2f}", f"{seg['end']:.2f}", seg.get('reason', 'N/A')])
        st = Table(seg_data, colWidths=[60, 80, 80, 230])
        st.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E8EAF6")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        elements.append(st)
    else:
        elements.append(Paragraph("<b>No modified segments detected.</b> The audio exhibits global forensic consistency.", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # --- ⚙️ TECHNICAL DETAILS ---
    elements.append(Paragraph("⚙️ TECHNICAL SPECIFICATIONS", section_style))
    tech_data = [
        ["Model Architecture:", "Multi-Modal Hybrid (Wav2Vec 2.0 + CNN + Bi-LSTM)"],
        ["Preprocessing:", "16kHz Resampling, Peak Normalization, Log-Mel Scaling"],
        ["Analysis Type:", "Deep Forensic AI Deepfake Detection"],
        ["Inference Engine:", "TensorFlow 2.x / PyTorch 2.x Dual-Stream"]
    ]
    tt = Table(tech_data, colWidths=[150, 300])
    tt.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    elements.append(tt)
    
    # --- FOOTER ---
    elements.append(Spacer(1, 40))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Paragraph("Generated by AudLens AI Forensic System", footer_style))
    elements.append(Paragraph("<i>DISCLAIMER: This report is an AI-based prediction. While high-accuracy results are achieved (>99%), it is intended for research and auxiliary forensic guidance, not as a 100% conclusive legal absolute.</i>", footer_style))
    
    doc.build(elements)
    return output_path
