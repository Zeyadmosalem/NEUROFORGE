from PIL import Image,ImageDraw,ImageFont,ImageFilter
from csp.generator import generate
import math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
W,H=1280,720
im=Image.new('RGB',(W,H));pixels=im.load()
for y in range(H):
    v=y/H
    for x in range(W):pixels[x,y]=(int(13+34*v),int(16+24*v),int(38+50*v))
d=ImageDraw.Draw(im)
r=random.Random(42)
for _ in range(110):
    x=r.randrange(W); y=r.randrange(10,290);s=r.choice([1,1,2]); d.ellipse((x,y,x+s,y+s),fill=(101,126,187))
for k,c in enumerate([(31,42,80),(34,49,87),(42,59,95)]):
    points=[(-20,H)]
    for x in range(-20,W+51,40):
        y=int(430+k*64-54*abs(math.sin((x+k*40)/160))-20*abs(math.sin((x+k*90)/79)))
        points.append((x,y))
    points.append((W+20,H));d.polygon(points,fill=c)
d.ellipse((1050,100,1150,200),fill=(100,76,145))
d.ellipse((1062,112,1138,188),fill=(186,110,174))
try:
    f1=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',40)
    f2=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    f3=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',21)
    f4=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
except OSError:f1=f2=f3=f4=ImageFont.load_default()
d.rounded_rectangle((24,20,1256,112),radius=20,fill=(22,28,56),outline=(72,88,129),width=2)
d.text((52,28),'N E U R O F O R G E',font=f1,fill=(104,230,239))
d.text((54,78),'THE PLATFORMER THAT LEARNS HOW YOU PLAY',font=f4,fill=(167,185,215))
d.text((710,39),'WORLD 03  /  AI GENERATED',font=f3,fill=(236,244,255))
d.text((710,74),'A* VERIFIED  •  AC-3 ENABLED',font=f4,fill=(135,229,179))
d.text((1050,36),'P(WIN)',font=f4,fill=(172,182,215))
d.text((1048,56),'71.9%',font=f1,fill=(140,237,182))
r=generate(seed=44,target=.7,n_chunks=7)
assert r.level
T=33;left=0;top=235
for y,row in enumerate(r.level.rows):
    for x,c in enumerate(row):
        if x>38:break
        sx=left+x*T; sy=top+y*T
        if c=='#':
            d.rectangle((sx,sy,sx+T,sy+T),fill=(49,43,85),outline=(79,76,115),width=1)
            if not r.level.solid(x,y-1):
                d.rounded_rectangle((sx,sy,sx+T,sy+7),radius=2,fill=(87,208,178))
                d.rectangle((sx,sy+7,sx+T,sy+11),fill=(36,116,122))
        elif c=='^':
            for a in (0,1):
                d.polygon([(sx+a*T/2+3,sy+T),(sx+(a+.5)*T/2,sy+3),(sx+(a+1)*T/2-3,sy+T)],fill=(249,99,152))
        elif c=='o':
            d.ellipse((sx+9,sy+8,sx+24,sy+23),outline=(248,200,102),width=4)
# Hero star-glow
x=150;y=top+9*T
light=Image.new('RGBA',(110,110),(0,0,0,0));g=ImageDraw.Draw(light)
g.ellipse((5,5,105,105),fill=(103,235,250,30))
light=light.filter(ImageFilter.GaussianBlur(10))
im.paste(light,(x-55,y-55),light)
d=ImageDraw.Draw(im)
d.rounded_rectangle((x-15,y-17,x+15,y+14),radius=7,fill=(37,49,102),outline=(108,234,240),width=3)
d.rounded_rectangle((x-11,y-12,x+11,y+1),radius=5,fill=(228,250,255))
d.ellipse((x-5,y-8,x-1,y-4),fill=(51,57,124));d.ellipse((x+3,y-8,x+7,y-4),fill=(51,57,124))
d.polygon([(x-15,y-10),(x-33,y+13),(x-15,y+12)],fill=(248,97,168))
# level marker/label; bottom panel
PANEL_TOP=H-64
d.rounded_rectangle((24,PANEL_TOP,1256,H-18),radius=12,fill=(19,25,53))
d.text((48,PANEL_TOP+15),'SEARCH  [ BFS  |  DFS  |  A*  |  GREEDY ]',font=f4,fill=(165,179,211))
d.text((555,PANEL_TOP+15),'CSP  [ BACKTRACK  |  FORWARD CHECK  |  AC-3 ]',font=f4,fill=(165,179,211))
d.text((1050,PANEL_TOP+15),'NEURAL  ON',font=f4,fill=(131,229,179))
d.text((48,145),'ONE SHARED PHYSICS ENGINE  /  ALL FOUR SEARCH POLICIES',font=f3,fill=(223,234,249))
d.text((48,176),'Adaptation pipeline: gameplay telemetry → neural prediction → CSP generation → A* verification',font=f2,fill=(166,189,213))
d.text((48,H-78),'OFFLINE LEVEL RENDER  •  illustrative interface preview, not an executed Pygame screenshot',font=f4,fill=(188,187,207))
out=ROOT/'assets'/'level_preview.png';im.save(out)
print(out)
