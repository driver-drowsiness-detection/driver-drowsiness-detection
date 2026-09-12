"""Generate editable SVG architecture figures for the planning guide."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent
INK = '#142c43'
MUTED = '#4b6379'
BLUE = '#2366a2'
TEAL = '#087f81'
PURPLE = '#7151a3'
ORANGE = '#a25e19'
LINE = '#7890a5'

class Figure:
    def __init__(self, name, height, number, title, subtitle):
        self.name = name
        self.h = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="{height}" viewBox="0 0 1500 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
            '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0 0 L9 4.5 L0 9Z" fill="#7890a5"/></marker></defs>',
            '<rect width="1500" height="100%" fill="#f7f9fc"/>',
            '<rect width="1500" height="12" fill="#142c43"/>']
        self.text(52, 62, f'MARVEL  /  DESIGN {number}', 16, TEAL, 600)
        self.text(52, 112, title, 35, INK, 700)
        self.text(52, 151, subtitle, 20, MUTED)

    def text(self,x,y,s,size=20,color=INK,weight=400,anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')

    def box(self,x,y,w,h,title,lines=(),color=BLUE,fill='#ffffff',tag=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#d6e0ea" stroke-width="1.4"/>')
        self.parts.append(f'<rect x="{x}" y="{y+16}" width="5" height="{h-32}" rx="2" fill="{color}"/>')
        yy=y+33
        if tag:
            self.text(x+23,yy,tag,14,color,700)
            yy+=33
        self.text(x+23,yy,title,23,INK,600)
        yy+=31
        for line in lines:
            self.text(x+23,yy,line,18,MUTED)
            yy+=27

    def area(self,x,y,w,h,label,color=BLUE,fill='#eef4fa'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}"/>')
        self.text(x+20,y+31,label,16,color,700)

    def path(self,points,dashed=False):
        d='M'+' L'.join(f'{x},{y}' for x,y in points)
        dash=' stroke-dasharray="7 6"' if dashed else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="2.3" stroke-linejoin="round"{dash} marker-end="url(#arrow)"/>')

    def label(self,x,y,s):
        self.text(x,y,s,16,MUTED,500)

    def finish(self,note):
        self.parts.append(f'<line x1="52" y1="{self.h-68}" x2="1448" y2="{self.h-68}" stroke="#d6e0ea"/>')
        self.text(52,self.h-35,note,17,MUTED)
        self.parts.append('</svg>')
        (OUT/f'{self.name}.svg').write_text('\n'.join(self.parts),encoding='utf-8')


f=Figure('01_entire_system',1510,'01','Entire project: learn, observe, decide, warn',
         'Training happens on Windows. Runtime is tested on Jetson. Speech never blocks the immediate alert.')
f.area(45,185,1410,245,'DEVELOPMENT  /  WINDOWS LAPTOPS',TEAL,'#e9f5f3')
f.box(70,245,390,145,'Labelled data', ['Subject groups and timestamps','Train / calibrate / validate / test'],TEAL)
f.box(540,245,390,145,'Learn and compare', ['Train CNN; map EAR scores','Evaluate EAR, CNN and fusion'],TEAL)
f.box(1010,245,420,145,'Versioned release bundle',['Weights + preprocessing + labels','Thresholds + alpha + checksums'],TEAL)
f.path([(460,315),(535,315)])
f.path([(930,315),(1005,315)])
f.area(45,470,400,600,'PERCEPTION  /  PHASE 1',BLUE)
f.area(470,470,480,600,'COGNITION  /  PHASE 1 + LATER',TEAL,'#e9f5f3')
f.area(975,470,480,600,'OUTPUT  /  PHASE 1',ORANGE,'#fcf3e9')
f.box(70,535,350,110,'Camera or replay',['Fresh frame + timestamp'])
f.box(70,695,350,145,'Face and eye localization',['Landmarks and bounded crops','Shared visibility checks'])
f.box(70,885,350,135,'Unusable observations',['UNKNOWN + diagnostic reason','Never invent open or closed'],ORANGE)
f.path([(245,645),(245,690)])
f.path([(245,840),(245,880)])
f.box(495,535,430,145,'Eye-state estimation',['EAR only / CNN only / combined','MAR measured as supporting data'],TEAL)
f.box(495,735,430,135,'Temporal interpretation',['Now: closure duration + recovery','Later: PERCLOS, pose, yawn cues'],TEAL)
f.box(495,915,430,105,'Alert decision',['Event ID, evidence and severity'],TEAL)
f.path([(420,765),(458,765),(458,605),(490,605)])
f.path([(710,680),(710,730)])
f.path([(710,870),(710,910)])
f.path([(1220,390),(1220,448),(710,448),(710,530)],True)
f.label(757,439,'Load fixed model and configuration')
f.box(1000,535,430,130,'Display and logs',['Measurements, frame age, errors','Run metadata and event records'],ORANGE)
f.box(1000,735,430,135,'Immediate fixed warning',['Sound or correctly driven buzzer','No dependency on language model'],ORANGE)
f.box(1000,915,430,105,'Bounded event queue',['Latest relevant event for speech'],PURPLE)
f.path([(925,967),(960,967),(960,802),(995,802)])
f.path([(925,977),(995,977)])
f.path([(925,605),(995,605)])
f.area(45,1110,1410,235,'SPEECH WORKER  /  LATER PHASE',PURPLE,'#f1edf8')
f.box(70,1170,390,135,'Local LLM',['Event summary → short sentence','Model and runtime still to verify'],PURPLE)
f.box(540,1170,390,135,'Validate or fall back',['Timeout, stale event, bad output','Use fixed text when appropriate'],PURPLE)
f.box(1010,1170,420,135,'Text to speech + speaker',['Play current, relevant warning','Record completion or failure'],PURPLE)
f.path([(1215,1020),(1215,1088),(440,1088),(440,1165)])
f.path([(460,1237),(535,1237)])
f.path([(930,1237),(1005,1237)])
f.text(70,1390,'Final evaluation: accuracy + coverage + false alerts + delay + memory under the actual camera conditions.',21,INK,500)
f.finish('Design specification, not measured performance. A combined experiment is required; improvement must be demonstrated.')


f=Figure('02_phase1_runtime',1610,'02','Phase 1: three methods, one fair comparison',
         'EAR and CNN are implemented separately. Combined mode uses comparable scores from the same eye and timestamp.')
f.box(360,200,780,110,'Input adapter',['Windows webcam / recorded video / verified Jetson CSI pipeline'],BLUE)
f.path([(750,310),(750,345)])
f.box(360,350,780,135,'Perception and quality',['Locate the driver; extract named points and eye crops','Validate geometry and crops separately; preserve frame identity'],BLUE)
f.box(55,355,255,130,'Invalid input',['UNKNOWN + reason','No fabricated value'],ORANGE)
f.path([(360,415),(315,415)])
f.path([(750,485),(750,525),(335,525),(335,560)])
f.path([(750,485),(750,525),(1165,525),(1165,560)])
f.box(65,565,540,160,'EAR branch',['Pixel landmarks → eye aspect ratio','Keep raw EAR for the standalone rule','Map EAR to a closure score for fusion'],TEAL,tag='GEOMETRY')
f.box(895,565,540,160,'CNN branch',['Eye crop → stored input transform','Trained CNN → closure score','Apply saved calibration if selected'],PURPLE,tag='LEARNED APPEARANCE')
f.path([(335,725),(335,815)])
f.path([(1165,725),(1165,815)])
f.path([(605,655),(690,655),(690,815)])
f.path([(895,655),(810,655),(810,815)])
f.text(750,760,'Same eye',16,MUTED,500,'middle')
f.text(750,785,'Same time',16,MUTED,500,'middle')
f.box(55,820,420,155,'EAR-only decision',['Raw EAR + selected threshold','Retain validity and eye identity'],TEAL,tag='MODE A')
f.box(540,820,420,155,'Combined decision',['alpha × CNN + (1 − alpha) × EAR','Uses mapped scores; both valid'],BLUE,tag='MODE C')
f.box(1025,820,420,155,'CNN-only decision',['CNN score + selected threshold','Retain validity and eye identity'],PURPLE,tag='MODE B')
for x in [265,750,1235]:
    f.path([(x,975),(x,1020)])
f.box(55,1025,420,165,'EAR temporal tracker',['Aggregate valid left/right states','Track closure and recovery','Emit method-specific events'],TEAL)
f.box(540,1025,420,165,'Combined temporal tracker',['Same aggregation and timing rules','Separate internal state','No silent fallback in first version'],BLUE)
f.box(1025,1025,420,165,'CNN temporal tracker',['Same aggregation and timing rules','Separate internal state','Record gaps and unknown periods'],PURPLE)
for x in [265,750,1235]:
    f.path([(x,1190),(x,1250),(750,1250),(750,1285)])
f.box(360,1290,780,130,'Compare, display and record',['EAR / CNN / fusion scores, events, quality and processing times','One selected sound output; separate logs for all three methods'],ORANGE)
f.label(70,1455,'Same controller code and timing configuration; one instance per method.')
f.text(70,1500,'CNN and fusion training are required in Phase 1. Full PERCLOS and LLM speech follow later.',22,INK,500)
f.finish('UNKNOWN does not prove recovery. Long gaps break confirmed closure continuity; an earlier alarm remains in history.')


f=Figure('03_training_evaluation',1460,'03','Training and testing without leaking answers',
         'Split by person and keep related recordings together. Never tune the combination on final-test outcomes.')
f.box(330,200,840,130,'Audit the dataset before training',['Full-frame paired evaluation + eye labels + timestamps + subject IDs','Eye-only training data may be useful, but may not support the EAR baseline'],BLUE)
f.path([(750,330),(750,365)])
f.box(330,370,840,110,'Freeze a grouped split manifest',['Parent frames, both eyes, crops and augmentations inherit the same partition'],BLUE)
xs=[55,425,795,1165]
centers=[220,590,960,1330]
for c in centers:
    f.path([(750,480),(750,520),(c,520),(c,555)])
f.box(xs[0],560,330,180,'Training',['Learn CNN weights','Augmentation here only','Save candidate checkpoints'],TEAL,tag='LEARN PARAMETERS')
f.box(xs[1],560,330,180,'Calibration',['Fit EAR score mapping','Check CNN probabilities','Keep subjects separate'],PURPLE,tag='MAKE SCORES COMPARABLE')
f.box(xs[2],560,330,180,'Validation',['Choose model and alpha','Choose operating thresholds','Choose temporal settings'],ORANGE,tag='SELECT SETTINGS')
f.box(xs[3],560,330,180,'Final test',['Keep labels untouched','Evaluate frozen choices','No tuning from outcomes'],BLUE,tag='MEASURE GENERALIZATION')
f.area(45,800,1410,250,'DEVELOPMENT OUTPUTS  /  FROM TRAINING, CALIBRATION AND VALIDATION ONLY',TEAL,'#e9f5f3')
f.box(75,865,390,140,'Model artefacts',['Selected CNN checkpoint','Architecture and class mapping'],TEAL)
f.box(555,865,390,140,'Decision artefacts',['EAR map, optional CNN calibration','Alpha, thresholds and timing rules'],TEAL)
f.box(1035,865,390,140,'Frozen release bundle',['Input transform and checksums','Versions, split ID, hardware target'],TEAL)
f.path([(220,740),(220,860)])
f.path([(590,740),(590,770),(750,770),(750,860)])
f.path([(960,740),(960,770),(750,770)])
f.path([(465,935),(550,935)])
f.path([(945,935),(1030,935)])
f.box(290,1120,920,160,'Final comparison: EAR vs CNN vs combined',['Same held-out observations and labelled videos; identical temporal policy','Report eye-state quality, observation coverage, event misses and false alerts','Also report Jetson latency, frame age, memory and difficult-condition slices'],BLUE)
f.path([(1230,1005),(1230,1080),(940,1080),(940,1115)])
f.path([(1330,740),(1470,740),(1470,1085),(1100,1085),(1100,1115)])
f.label(1240,1070,'Test data')
f.text(75,1350,'Fusion can lose. Preserve all three results and explain the tradeoff instead of forcing a win.',23,INK,500)
f.finish('If subjects are too few for these partitions, use grouped development folds and an untouched outer test; report limitations.')

print('Created 3 SVG diagrams')
