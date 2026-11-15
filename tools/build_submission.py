"""Regenerate edited 12-page-max report .docx and presentation .pptx.
Requires python-docx + python-pptx (optional submission tooling).
"""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs'

def build_report():
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc=Document()
    sec=doc.sections[0]
    sec.top_margin=Inches(.72);sec.bottom_margin=Inches(.69)
    sec.left_margin=Inches(.80);sec.right_margin=Inches(.80)
    styles=doc.styles
    styles['Normal'].font.name='Aptos';styles['Normal'].font.size=Pt(9.1);styles['Normal'].font.color.rgb=RGBColor(39,48,67)
    styles['Normal'].paragraph_format.space_after=Pt(4.7)
    styles['Normal'].paragraph_format.line_spacing=1.10
    for s,size,space in [('Title',19,12),('Heading 1',13,8),('Heading 2',10.5,5)]:
        st=styles[s];st.font.name='Aptos Display';st.font.size=Pt(size);st.font.bold=True
        st.font.color.rgb=RGBColor(39,77,128)
        st.paragraph_format.space_before=Pt(space)
        st.paragraph_format.space_after=Pt(5)
    # Footer with page field
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('NEUROFORGE  ·  TECHNICAL REPORT  |  ').font.size=Pt(8)
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
    lines=(DOC/'REPORT.md').read_text(encoding='utf-8').splitlines()
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line.startswith('|'):
            group=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                group.append([v.strip() for v in lines[i].strip().strip('|').split('|')]);i+=1
            clean=[r for r in group if not all(re.fullmatch(r':?-{3,}:?',c) for c in r)]
            if clean:
                table=doc.add_table(rows=1,cols=len(clean[0]))
                table.style='Light Shading Accent 1';table.alignment=WD_TABLE_ALIGNMENT.CENTER
                for j,text in enumerate(clean[0]): table.rows[0].cells[j].text=text
                for row in clean[1:]:
                    cells=table.add_row().cells
                    for j,text in enumerate(row): cells[j].text=text.replace('**','')
                for rowidx,row in enumerate(table.rows):
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            for run in p.runs:run.font.size=Pt(7.7)
                doc.add_paragraph('')
            continue
        if line.startswith('!['):
            found=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            if found:
                p=ROOT/(DOC/found[2]).resolve().relative_to(ROOT)
                if p.exists():
                    par=doc.add_paragraph();par.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    par.add_run().add_picture(str(p),width=Inches(5.35))
            i+=1;continue
        if line.startswith('# '):
            doc.add_paragraph(line[2:],style='Title');i+=1;continue
        if line.startswith('### '):doc.add_paragraph(line[4:],style='Heading 2');i+=1;continue
        if line.startswith('## '):doc.add_paragraph(line[3:],style='Heading 1');i+=1;continue
        par=doc.add_paragraph()
        # lightweight bold/italic processing
        parts=re.split(r'(\*\*.*?\*\*)',line)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                run=par.add_run(part[2:-2]);run.bold=True
            else:
                p=part.replace('`','').replace('*','')
                par.add_run(p)
        i+=1
    out=DOC/'NEUROFORGE_Technical_Report.docx'
    doc.save(out);print('Report:',out)
    return out


def build_slides():
    from pptx import Presentation
    from pptx.util import Inches,Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.enum.shapes import MSO_SHAPE
    p=Presentation();p.slide_width=Inches(13.333);p.slide_height=Inches(7.5)
    NAVY=RGBColor(13,17,38);PANEL=RGBColor(28,35,69);WHITE=RGBColor(233,241,254)
    MUTED=RGBColor(166,182,211);CYAN=RGBColor(100,233,246);GOLD=RGBColor(255,202,112)
    PINK=RGBColor(255,112,181);GREEN=RGBColor(132,235,180)
    def block(sl,x,y,w,h,color,radius=True):
        sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                               Inches(x),Inches(y),Inches(w),Inches(h))
        sh.fill.solid();sh.fill.fore_color.rgb=color
        sh.line.fill.background()
        return sh
    def tx(sl,x,y,w,h,text,size=17,color=WHITE,bold=False,align=None):
        sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
        tf=sh.text_frame;tf.word_wrap=True
        tf.margin_left=Inches(.05);tf.margin_top=Inches(.02)
        for j,line in enumerate(text.split('\n')):
            q=tf.paragraphs[0] if j==0 else tf.add_paragraph()
            q.text=line;q.font.name='Aptos';q.font.size=Pt(size);q.font.bold=bold;q.font.color.rgb=color
            if align:q.alignment=align
            q.space_after=Pt(7)
        return sh
    def slide(title,kicker='NEUROFORGE  /  AI SYSTEMS PROJECT'):
        s=p.slides.add_slide(p.slide_layouts[6])
        block(s,0,0,13.333,7.5,NAVY,False)
        block(s,0,.0,13.333,.085,CYAN,False)
        tx(s,.7,.4,11,.35,kicker,size=12,color=CYAN,bold=True)
        tx(s,.7,.9,12,1.1,title,size=33,bold=True)
        tx(s,.72,7.06,10,.25,'SEARCH + CSP + NEURAL NETWORK  •  REPRODUCIBLE DEMONSTRATION',size=10,color=MUTED)
        tx(s,12.18,7.04,.45,.25,f'{len(p.slides):02d}',size=11,color=CYAN)
        return s
    def item(s,x,y,w,num,heading,body,accent=CYAN):
        block(s,x,y,w,1.55,PANEL)
        tx(s,x+.2,y+.16,.5,.4,num,size=20,color=accent,bold=True)
        tx(s,x+.8,y+.12,w-.96,.44,heading,size=20,bold=True)
        tx(s,x+.22,y+.65,w-.42,.76,body,size=15,color=MUTED)
    # cover
    s=slide('THE PLATFORMER THAT LEARNS HOW YOU PLAY','NEUROFORGE  /  PROJECT OVERVIEW')
    s.shapes.add_picture(str(ROOT/'assets'/'level_preview_banner.png'),Inches(.7),Inches(2.00),width=Inches(11.93))
    # 2
    s=slide('One game. Three AI paradigms.')
    item(s,.7,2.10,3.85,'01','NEURAL NETWORK','Estimates player–level completion probability.',CYAN)
    item(s,4.76,2.10,3.85,'02','CONSTRAINT SOLVER','Builds challenge-matched levels with hard rules.',GOLD)
    item(s,8.82,2.10,3.85,'03','GRAPH SEARCH','Verifies playability; compares BFS, DFS, A* and Greedy.',GREEN)
    tx(s,1.15,4.38,11.2,1.3,'Research question: can player-aware candidate selection generate verified levels while making every AI decision inspectable?',size=25,color=WHITE,align=PP_ALIGN.CENTER)
    # 3 pipeline
    s=slide('A closed adaptation loop')
    cards=[('1','PLAY','Record attempts and jump outcomes'),('2','PREDICT','NN estimates P(win | profile, level)'),
           ('3','CONSTRAIN','CSP assembles valid chunks'),('4','VERIFY','A* tests physics-based reachability'),('5','ADAPT','Choose the next appropriate level')]
    for i,(n,a,b) in enumerate(cards):
        x=.55+i*2.58
        block(s,x,2.38,2.39,2.05,PANEL)
        tx(s,x+.18,2.62,.55,.5,n,size=30,color=[CYAN,GOLD,PINK,GREEN,CYAN][i],bold=True)
        tx(s,x+.18,3.2,2.0,.4,a,size=18,bold=True)
        tx(s,x+.18,3.68,2.03,.58,b,size=12,color=MUTED)
        if i<4: tx(s,x+2.37,3.18,.35,.45,'→',size=26,color=CYAN)
    tx(s,1.14,5.23,11,.77,'Each candidate is generated by CSP, physically verified by search, then scored against the player model.',size=18,color=MUTED,align=PP_ALIGN.CENTER)
    # 4
    s=slide('Search uses the actual movement physics')
    tx(s,.8,2.12,11.5,.55,'STATE  =  (x, y, vertical velocity, world phase)       ACTION COST  =  1 logical tick',size=19,color=CYAN,bold=True)
    item(s,.7,2.85,5.8,'B','BFS  /  DFS','Queue vs stack. BFS guarantees shortest path; DFS does not.',GOLD)
    item(s,6.73,2.85,5.8,'A','A*  /  GREEDY','Both use distance-to-goal h; A* also uses accumulated cost.',GREEN)
    tx(s,.85,5.2,11.9,.8,'Visible counters: expanded / generated states, path cost, runtime, search frontier and route trace.',size=19,color=MUTED)
    # 5
    s=slide('CSP creates meaningful, valid levels')
    item(s,.7,2.22,3.85,'B','BACKTRACKING','Assign a chunk; reject inconsistent choices; backtrack.',CYAN)
    item(s,4.76,2.22,3.85,'F','FORWARD CHECK','Prune incompatible neighbor-domain values.',GOLD)
    item(s,8.82,2.22,3.85,'A','AC-3','Propagate arc consistency across binary constraints.',PINK)
    tx(s,1.0,4.80,11.4,1.15,'Hard design rules: safe start and goal, compatible neighbors, pacing, variety and difficulty budget. A* performs the final reachability check.',size=20,color=MUTED,align=PP_ALIGN.CENTER)
    # 6
    s=slide('A genuine neural network — and a fair baseline')
    block(s,.7,2.15,4.4,3.4,PANEL)
    tx(s,1.0,2.44,3.9,.75,'12  →  32  →  16  →  1',size=27,color=CYAN,bold=True)
    tx(s,1.0,3.35,3.85,1.55,'NumPy MLP • ReLU hidden layers\nAdam • binary cross-entropy\nPlayer + level features',size=17)
    block(s,5.38,2.15,7.25,3.4,PANEL)
    tx(s,5.73,2.40,6.7,.52,'988 held-out synthetic rows',size=20,color=GOLD,bold=True)
    tx(s,5.73,3.16,6.5,1.9,'NN accuracy: 69.1%  |  AUC: 0.740\nLogistic accuracy: 74.3%  |  AUC: 0.798\n\nThe baseline wins. No human validation is claimed.',size=18)
    tx(s,.78,5.94,11.6,.45,'DATA PROVENANCE: 6,500 simulated attempts; group-disjoint simulated players; VGLC geometry downloaded separately.',size=13,color=MUTED)
    # 7
    s=slide('Experiment 1  /  four search strategies')
    block(s,.65,2.0,8.15,4.52,RGBColor(247,249,252))
    s.shapes.add_picture(str(ROOT/'results'/'search_plot.png'),Inches(.79),Inches(2.19),width=Inches(7.88))
    tx(s,9.12,2.10,3.4,1.15,'5 seeds',size=29,color=CYAN,bold=True)
    tx(s,9.12,3.0,3.4,3.4,'BFS: 4480 expanded\nA*: 502 expanded\nGreedy: 59 expanded\nDFS: 56 expanded\n\nBFS and A* are optimal (cost 53.6); DFS paths are longer (54.6).',size=18)
    # 8
    s=slide('Experiment 2  /  switchable CSP pruning')
    block(s,.65,2.0,8.15,4.52,RGBColor(247,249,252))
    s.shapes.add_picture(str(ROOT/'results'/'csp_plot.png'),Inches(.79),Inches(2.19),width=Inches(7.88))
    tx(s,9.12,2.10,3.4,.7,'5 seeds',size=29,color=CYAN,bold=True)
    tx(s,9.12,3.03,3.45,3.0,'Backtracking: 6.4 backtracks\nFC: 10.6\nAC-3: 10.6\nBoth: 10.6\n\nFC / AC-3 prune ~12 values each but did not cut backtracks here; SDs are as large as the means.',size=17)
    # 9
    s=slide('Experiment 3  /  synthetic adaptation')
    block(s,.65,2.0,8.15,4.52,RGBColor(247,249,252))
    s.shapes.add_picture(str(ROOT/'results'/'adaptation_plot.png'),Inches(.79),Inches(2.19),width=Inches(7.88))
    tx(s,9.12,2.13,3.4,.9,'200 per method',size=23,color=CYAN,bold=True)
    tx(s,9.12,3.02,3.4,3.2,'Fixed: 58.0% win\nRule: 72.5% win\nNeural: 72.5% win\n\nTarget is 70%. These are simulated outcomes, not human win rates.',size=17)
    # 10
    s=slide('Live demo  /  a five-minute checklist')
    demos=[('01','Play a seeded level','Actual movement, telemetry and adaptivity'),
           ('02','Press TAB; try 1–4','Search traces, paths, counters'),
           ('03','Toggle F/C then G','Backtracking vs Forward Checking / AC-3'),
           ('04','Toggle N','Neural model vs rule-based baseline'),
           ('05','Run headless demo','Verify impossible CSP → “No solution”')]
    for i,(n,a,b) in enumerate(demos):
        x=.9+(i%2)*6.0;y=2.05+(i//2)*1.46
        block(s,x,y,5.7,1.25,PANEL)
        tx(s,x+.15,y+.17,.7,.54,n,size=21,color=CYAN,bold=True)
        tx(s,x+.86,y+.10,4.6,.47,a,size=20,bold=True)
        tx(s,x+.87,y+.66,4.55,.4,b,size=13,color=MUTED)
    # 11
    s=slide('What is complete — and what comes next')
    item(s,.8,2.1,5.7,'✓','IMPLEMENTED','All four search algorithms, CSP modes, NN, integrated demo, 44 passing tests, five-seed experiments.',GREEN)
    item(s,6.86,2.1,5.7,'!','LIMITATIONS','Synthetic player outcomes, NN weaker than the baseline, simple discrete physics, finite chunk catalog.',GOLD)
    tx(s,.9,4.40,11.5,1.7,'Next: human play-testing, harder CSP/search instances, improved calibration and richer physics.\n\nScience means reporting mixed results—not inventing model improvements.',size=20,color=MUTED,align=PP_ALIGN.CENTER)
    out=DOC/'NEUROFORGE_Presentation.pptx'
    p.save(out);print('Presentation:',out)
    return out

if __name__=='__main__':
    build_report()
    build_slides()
