"""Offline preview of the four independent generated worlds (no pygame needed).

This is a styled preview using the *real* states and maps; NOT a screenshot of a
Pygame runtime. The actual arena GUI uses game/arena_app.py.
"""
from __future__ import annotations
from random import Random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from integration.arena import make_arena, STYLE

ROOT=Path(__file__).resolve().parent
W,H=1372,926
TILE=17
PALETTE={ 'bfs':(89,218,236),'dfs':(181,121,248),
          'astar':(111,237,170),'greedy':(255,117,164)}
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
REG='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=lambda n,bold=False:ImageFont.truetype(BOLD if bold else REG,n)
F={'logo':font(35,True),'heading':font(22,True),'sub':font(15,True),
   'body':font(13),'small':font(11),'micro':font(9)}

def make(out=ROOT/'assets'/'arena_preview.png',seed=42):
    lanes=make_arena(seed, persist=False)
    for i,a in enumerate(lanes):
        for t in range(235+i*17):a.tick()
    im=Image.new('RGB',(W,H),(10,13,30));d=ImageDraw.Draw(im)
    r=Random(9)
    for _ in range(190):
        x=r.randrange(W);y=r.randrange(H);v=r.choice([30,35,40,50])
        d.ellipse((x,y,x+2,y+2),fill=(v,v+10,v+30))
    d.text((25,12),'NEUROFORGE',fill=(119,226,239),font=F['logo'])
    d.text((25,62),'V3: REACTIVE WORLD SURPRISES   /   FOUR EVOLVING WORLDS',fill=(153,168,197),font=F['sub'])
    d.text((1020,31),'AUTONOMOUS ARENA',fill=(222,232,244),font=F['heading'])
    d.line((18,92,W-18,92),fill=(55,72,106),width=1)
    cw=(W-36-14)//2;ch=(H-100-56-14-18)//2
    for j,a in enumerate(lanes):
        x0=18+(j%2)*(cw+14);y0=100+(j//2)*(ch+14)
        color=PALETTE[a.algorithm]
        d.rounded_rectangle((x0,y0,x0+cw,y0+ch),14,fill=(20,25,49),outline=(65,73,107),width=1)
        d.rounded_rectangle((x0,y0,x0+cw,y0+64),12,fill=tuple(min(255,int(v*.20+26)) for v in color))
        d.rectangle((x0,y0,x0+5,y0+ch),fill=color)
        d.text((x0+17,y0+10),STYLE[a.algorithm]['title'],fill=color,font=F['heading'])
        d.text((x0+17,y0+43),
               {'bfs':'EXPLORES THE WHOLE FRONTIER','dfs':'CHASES DEEP BRANCHES',
                'astar':'WEIGHS COST + DISTANCE','greedy':'CHASES THE GOAL'}[a.algorithm],
               fill=(151,162,194),font=F['small'])
        d.text((x0+cw-99,y0+15),a.algorithm.upper(),fill=(236,244,255),font=F['sub'])
        wy=y0+65
        d.rounded_rectangle((x0+7,wy,x0+cw-7,wy+12*TILE),8,fill=(13,23,46))
        cam=max(0,min(max(0,a.level.width*TILE-(cw-24)),a.state.x*TILE-(cw-24)*.35))
        for k,chunk in enumerate(a.level.chunks):
            px=x0+12+k*8*TILE-cam
            if x0<=px<=x0+cw-30:
                d.line((px,wy,px,wy+12*TILE),fill=(36,55,80),width=1)
                d.text((px+4,wy+5),chunk[:10].upper(),fill=(79,110,139),font=F['micro'])
        for x in range(a.level.width):
            sx=int(x0+12+x*TILE-cam)
            if sx<x0+7 or sx>x0+cw-24:continue
            for y in range(a.level.height):
                yy=wy+y*TILE;tile=a.level.tile(x,y)
                if tile=='#':
                    d.rectangle((sx,yy,sx+TILE-1,yy+TILE-1),fill=(44,45,79))
                    if not a.level.solid(x,y-1):
                        d.rectangle((sx,yy,sx+TILE-1,yy+3),fill=(66,174,155))
                elif tile=='^':
                    for k in (0,1):
                        xx=sx+k*TILE/2
                        d.polygon([(xx,yy+TILE),(xx+TILE/4,yy+2),
                                   (xx+TILE/2,yy+TILE)],fill=(240,91,134))
                elif tile=='o':
                    d.ellipse((sx+5,yy+5,sx+11,yy+11),outline=(255,210,107),width=2)
        for platform in a.level.platforms:
            xx,yy=platform.position(a.state.phase)
            sx=int(x0+12+xx*TILE-cam);sy=int(wy+yy*TILE)
            d.rounded_rectangle((sx,sy,sx+platform.width*TILE,sy+8),3,
                                fill=(104,216,220),outline=(220,246,250),width=1)
        for enemy in a.level.enemies:
            ex,ey=enemy.position(a.state.phase)
            sx=int(x0+12+ex*TILE-cam+TILE//2)
            sy=int(wy+ey*TILE+TILE//2)
            d.ellipse((sx-7,sy-7,sx+7,sy+7),fill=(238,83,127))
            d.ellipse((sx-3,sy-3,sx+1,sy+1),fill=(253,237,230))
        for gate in a.level.gates:
            gx=int(x0+12+gate.x*TILE-cam+TILE//2)
            active=gate.active(a.state.phase)
            d.line((gx,wy+gate.y_top*TILE,gx,wy+(gate.y_bottom+1)*TILE),
                   fill=(255,86,142) if active else (74,110,147),width=4 if active else 2)
        if a.plan.found:
            pts=[(int(x0+12+s.x*TILE-cam+TILE//2),
                  int(wy+(s.y-1)*TILE+TILE//2)) for s in a.plan.path[a.plan_index:]]
            if len(pts)>1:d.line(pts,fill=color,width=2)
        px=x0+12+a.state.x*TILE-cam+TILE//2;py=wy+(a.state.y-1)*TILE+TILE//2
        d.ellipse((px-14,py-14,px+14,py+14),outline=color,width=3)
        d.rounded_rectangle((px-8,py-10,px+8,py+10),4,fill=(24,33,69),outline=color,width=2)
        d.rectangle((px-5,py-3,px+5,py+3),fill=(230,250,255))
        info=wy+12*TILE+10
        d.rounded_rectangle((x0+14,info,x0+cw-15,info+6),3,fill=(32,42,66))
        d.rounded_rectangle((x0+14,info,x0+14+max(3,int((cw-29)*a.state.x/max(1,a.level.goal_x))),info+6),3,fill=color)
        d.text((x0+15,info+15),f'W {a.wins:02}    L {a.losses:02}',fill=(245,247,255),font=F['sub'])
        d.text((x0+330,info+15),f'DIFF {a.level.difficulty:.2f}   TARGET {a.controller.challenge:.2f}',fill=color,font=F['body'])
        d.text((x0+15,info+37),f'EXP {a.plan.expanded}   REPLANS {a.replans}   EVENTS {a.total_surprises}',fill=(153,165,194),font=F['small'])
        d.text((x0+330,info+37),f'P(WIN) {a.choice.predicted_win:.0%}',fill=(238,245,255),font=F['small'])
        d.text((x0+15,info+56),f'ONLINE UPDATE {a.controller.adapter.updates}  /  FRESH {len(a.controller.adapter.pending)}/10',fill=color,font=F['body'])
        d.text((x0+330,info+56),f'NEXT MEMORY DROP {30-a.controller.adapter.observations%30}',
               fill=(255,181,125),font=F['small'])
        d.text((x0+15,info+74),'SEEN: '+(a.world_events[-1]['description'] if a.world_events else 'OBSERVING A CHANGING WORLD'),
               fill=(142,158,189),font=F['small'])
    footer=(17,H-56+6,W-18,H-9)
    d.rounded_rectangle(footer,9,fill=(18,23,46))
    d.text((28,H-56+17),'SPACE PAUSE    +/- SPEED    H ROUTES    U CHAOS ON/OFF    E SURPRISE NOW    X MEMORY GLITCH',
           fill=(233,240,255),font=F['body'])
    d.text((1006,H-56+18),'M  MANUAL GAME',fill=(115,232,183),font=F['sub'])
    out=Path(out);out.parent.mkdir(exist_ok=True,parents=True)
    im.save(out)
    return out
if __name__=='__main__':print(make())
