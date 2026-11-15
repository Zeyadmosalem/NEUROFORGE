"""NEUROFORGE: original neon-platformer UI, built with Pygame CE.
ALL gameplay movement is resolved by game.physics.step, the same as search.
The game has zero imported Nintendo art. All shapes are code-drawn.
"""
from __future__ import annotations
from math import sin,cos,pi
from time import monotonic
from random import Random
from game.physics import step, State, is_dead
from search.algorithms import solve
from integration.adaptive import AdaptiveController

WIDTH,HEIGHT=1120,650
TILE=35
TOP=130
TICK=0.105
COLORS={'bg':(14,16,37),'panel':(23,28,57),'pink':(255,101,177),
        'cyan':(103,232,245),'gold':(255,203,109),'green':(129,231,171),
        'white':(232,243,255),'muted':(152,163,195),'purple':(139,115,231)}

def run(seed=42):
    try:
        import pygame as pg
    except ImportError as ex:
        raise RuntimeError('Install GUI dependency with: pip install pygame-ce==2.5.8') from ex
    pg.init()
    try:
        screen=pg.display.set_mode((WIDTH,HEIGHT))
        pg.display.set_caption('NEUROFORGE  //  Adaptive AI Platformer')
        clock=pg.time.Clock()
        fonts={'title':pg.font.SysFont('dejavusans',36,bold=True),
               'sub':pg.font.SysFont('dejavusans',20,bold=True),
               'body':pg.font.SysFont('dejavusans',16),
               'small':pg.font.SysFont('dejavusans',12)}
        controller=AdaptiveController(seed=seed,owner='human',persist=True)
        chosen=controller.next_level(n_candidates=5)
        level=chosen.result.level
        state=level.start;prev=state
        display_status='READY';status_until=monotonic()+2
        grounded_last=True; jumps=0;landings=0;attempt_start=monotonic();deaths=0
        stage=1;tokens=set();score=0
        enable_fc=True;enable_ac=True; ai_lab=False;paused=False
        lab_algo='astar';lab_results={};lab_t=0;lab_auto=False
        camera=0.;animation_time=0.;sim_acc=0.;game_start=monotonic()
        stars=[(Random(i).random()*WIDTH,Random(i+500).random()*TOP,Random(i+200).random()*1.6+0.3)
               for i in range(90)]

        def text(s,x,y,color='white',font='body',center=False):
            surf=fonts[font].render(str(s),True,COLORS.get(color,color))
            screen.blit(surf,(x-surf.get_width()/2 if center else x,y))

        def rounded(r,color,radius=12,width=0):
            pg.draw.rect(screen,color,pg.Rect(r),border_radius=radius,width=width)

        def fresh_level():
            nonlocal level,chosen,state,prev,stage,attempt_start,jumps,landings,grounded_last
            nonlocal camera,sim_acc,lab_results,lab_t,tokens,display_status,status_until
            chosen=controller.next_level(n_candidates=5)
            level=chosen.result.level
            state=level.start;prev=state;stage+=1
            attempt_start=monotonic();jumps=0;landings=0;grounded_last=True
            camera=0;sim_acc=0;lab_results={};lab_t=0;tokens=set()
            display_status='AI BUILT THE NEXT CHALLENGE';status_until=monotonic()+2.8

        def reset_state():
            nonlocal state,prev,jumps,landings,attempt_start,grounded_last
            state=level.start;prev=state;jumps=0;landings=0
            attempt_start=monotonic();grounded_last=True

        def get_lab_result(algo):
            if algo not in lab_results:lab_results[algo]=solve(level,algo,max_expansions=18000)
            return lab_results[algo]

        def draw_background(t):
            screen.fill(COLORS['bg'])
            for sy in range(0,HEIGHT,3):
                blend=sy/HEIGHT
                pg.draw.rect(screen,(int(14+18*blend),int(16+21*blend),int(37+39*blend)),(0,sy,WIDTH,3))
            for sx,sy,r in stars:
                twinkle=0.6+0.4*sin(t*1.4+sx)
                pg.draw.circle(screen,(int(65+80*twinkle),int(96+80*twinkle),int(135+70*twinkle)),
                               (int(sx-camera*0.07)%WIDTH,int(sy+20)),max(1,int(r)))
            for layer in range(3):
                rate=0.09+layer*0.13
                base=HEIGHT-180+layer*55
                col=[(28,35,72),(35,43,82),(44,57,92)][layer]
                coords=[(-30,HEIGHT)]
                for x in range(-30,WIDTH+110,65):
                    yy=base-abs(sin((x+camera*rate)*0.005+layer))*50-abs(sin((x+camera*rate)*0.013))*20
                    coords.append((x,int(yy)))
                coords.append((WIDTH+30,HEIGHT))
                pg.draw.polygon(screen,col,coords)
            pg.draw.circle(screen,(113,80,168),(970-int(camera*0.025),155),70)
            pg.draw.circle(screen,(189,111,173),(970-int(camera*0.025),155),51)

        def draw_world(t,alpha):
            nonlocal camera
            px=prev.x+(state.x-prev.x)*alpha
            py=prev.y+(state.y-prev.y)*alpha
            desired=max(0,px*TILE-WIDTH*0.40)
            camera+=(desired-camera)*0.12
            for y,row in enumerate(level.rows):
                if y>11:continue
                for x,tile in enumerate(row):
                    if tile=='.':continue
                    xx=x*TILE-camera;yy=TOP+y*TILE
                    if xx<-TILE or xx>WIDTH+TILE:continue
                    if tile=='#':
                        pg.draw.rect(screen,(44,42,79),(int(xx),int(yy),TILE,TILE))
                        edge=not level.solid(x,y-1)
                        if edge:
                            pg.draw.rect(screen,(64,188,169),(int(xx),int(yy),TILE,7),border_radius=3)
                            pg.draw.rect(screen,(32,114,119),(int(xx),int(yy+7),TILE,4))
                        pg.draw.line(screen,(77,68,112),(int(xx+4),int(yy+TILE-4)),
                                     (int(xx+TILE-6),int(yy+TILE-4)),1)
                    elif tile=='^':
                        for k in range(2):
                            points=[(int(xx+k*TILE/2+2),int(yy+TILE)),
                                    (int(xx+k*TILE/2+TILE/4),int(yy+8)),
                                    (int(xx+(k+1)*TILE/2-2),int(yy+TILE))]
                            pg.draw.polygon(screen,(246,92,147),points)
                            pg.draw.line(screen,(255,194,212),points[0],points[1],2)
                    elif tile=='o':
                        if (x,y) not in tokens:
                            cy=yy+TILE//2+int(sin(t*4+x)*4)
                            pg.draw.circle(screen,(114,84,36),(int(xx+TILE/2),int(cy)),11)
                            pg.draw.circle(screen,COLORS['gold'],(int(xx+TILE/2),int(cy)),8,3)
            # Physically simulated moving platforms, patrolling enemies and
            # timed energy gates: the human sees the same world as the planners.
            phase=state.phase
            for platform in level.platforms:
                xx,yy=platform.position(phase)
                sx=int(xx*TILE-camera);sy=int(TOP+yy*TILE)
                pg.draw.rect(screen,(101,218,213),(sx,sy,platform.width*TILE,TILE//2),border_radius=5)
                pg.draw.rect(screen,(198,251,239),(sx+3,sy+2,platform.width*TILE-6,5),border_radius=2)
            for enemy in level.enemies:
                ex,ey=enemy.position(phase)
                sx=int(ex*TILE-camera+TILE/2);sy=int(TOP+ey*TILE+TILE/2)
                pg.draw.circle(screen,(246,100,138),(sx,sy),TILE//2-3)
                pg.draw.circle(screen,(250,235,232),(sx-5,sy-6),5)
                pg.draw.circle(screen,(26,29,67),(sx-4,sy-6),2)
            for gate in level.gates:
                gx=int(gate.x*TILE-camera+TILE//2)
                a=gate.active(phase)
                pg.draw.line(screen,(252,86,146) if a else (72,104,133),
                             (gx,TOP+gate.y_top*TILE),
                             (gx,TOP+(gate.y_bottom+1)*TILE),6 if a else 2)
            # portal finish
            goalx=level.goal_x*TILE-camera
            gy=TOP+9*TILE-24
            if -100<goalx<WIDTH+100:
                pr=17+int(3*sin(t*5))
                pg.draw.ellipse(screen,(44,122,125),(int(goalx-12),int(gy-36),40,96),5)
                pg.draw.ellipse(screen,COLORS['cyan'],(int(goalx-12),int(gy-36),40,96),3)
                pg.draw.circle(screen,(239,182,255),(int(goalx+7),int(gy+12)),pr,3)
            # player body: intentionally original shape, a neon explorer
            X=int(px*TILE-camera+TILE/2);Y=int(TOP+(py-1)*TILE+TILE/2)
            if -50<X<WIDTH+50:
                shadow=pg.Surface((58,58),pg.SRCALPHA)
                pg.draw.circle(shadow,(98,231,255,53),(29,29),28)
                screen.blit(shadow,(X-29,Y-30))
                pg.draw.rect(screen,(23,31,73),(X-12,Y-12,24,27),border_radius=8)
                pg.draw.rect(screen,COLORS['cyan'],(X-12,Y-12,24,27),border_radius=8,width=2)
                pg.draw.rect(screen,(224,249,253),(X-9,Y-7,18,10),border_radius=4)
                pg.draw.circle(screen,(35,54,104),(X-3,Y-2),2)
                pg.draw.circle(screen,(35,54,104),(X+4,Y-2),2)
                pg.draw.polygon(screen,COLORS['pink'],[(X-12,Y-5),(X-22,Y+9),(X-12,Y+13)])
                pg.draw.rect(screen,(88,84,189),(X-10,Y+12,8,7),border_radius=2)
                pg.draw.rect(screen,(88,84,189),(X+3,Y+12,8,7),border_radius=2)
            # trace, drawn after tiles but below HUD
            if ai_lab and lab_algo in lab_results:
                res=lab_results[lab_algo]
                count=min(len(res.trace),lab_t)
                for s in res.trace[:count]:
                    xx=int(s.x*TILE-camera+TILE/2)
                    yy=int(TOP+(s.y-1)*TILE+TILE/2)
                    if -20<xx<WIDTH+20:
                        pg.draw.circle(screen,COLORS['pink'],(xx,yy),3)
                if res.found:
                    points=[(int(s.x*TILE-camera+TILE/2),int(TOP+(s.y-1)*TILE+TILE/2)) for s in res.path]
                    if len(points)>1:pg.draw.lines(screen,COLORS['gold'],False,points,3)

        def draw_hud(t):
            rounded((15,12,WIDTH-30,108),(18,23,50),18)
            rounded((15,12,WIDTH-30,108),(66,79,125),18,1)
            text('N E U R O F O R G E',34,22,'cyan','title')
            text('THE PLATFORMER THAT LEARNS HOW YOU PLAY',37,68,'muted','small')
            text(f'WORLD {stage:02}',575,24,'white','sub')
            text(f'COINS  {score:03}',575,58,'gold','body')
            text(f'DEATHS  {deaths:02}',710,58,'pink','body')
            prob=chosen.predicted_win*100
            text('PREDICTED SUCCESS',832,26,'muted','small')
            text(f'{prob:04.1f}%',832,47,'green' if prob>=65 else 'gold','title')
            rounded((15,HEIGHT-57,WIDTH-30,45),(18,23,50),12)
            text('A/D OR ARROWS  MOVE    SPACE / W  JUMP    TAB  AI LAB    1-4  SEARCH',30,HEIGHT-45,'muted','small')
            text('F FC   C AC-3   N NN   G GENERATE   R RESET   V 4-AGENT ARENA   X MEMORY GLITCH',30,HEIGHT-28,'muted','small')
            if monotonic()<status_until:
                surf=pg.Surface((530,51),pg.SRCALPHA);surf.fill((22,28,58,230))
                screen.blit(surf,(WIDTH//2-265,TOP+9))
                text(display_status,WIDTH//2,TOP+21,'gold','sub',center=True)

        def draw_lab():
            nonlocal lab_t
            if not ai_lab:return
            overlay=pg.Surface((410,445),pg.SRCALPHA)
            overlay.fill((11,17,40,234))
            screen.blit(overlay,(WIDTH-425,TOP+17))
            X=WIDTH-405;Y=TOP+33
            text('AI LABORATORY',X,Y,'cyan','sub')
            text('Same physics graph, different search policies',X,Y+30,'muted','small')
            res=get_lab_result(lab_algo)
            lab_t=min(len(res.trace),lab_t+7)
            text('SEARCH',X,Y+66,'white','sub')
            text(f'ALGORITHM    {lab_algo.upper()}',X,Y+96,'gold','body')
            text(f'SOLVED       {res.found}',X,Y+120,'green' if res.found else 'pink','body')
            text(f'EXPANDED     {res.expanded}',X,Y+144,'white','body')
            text(f'GENERATED    {res.generated}',X,Y+168,'white','body')
            text(f'PATH COST    {res.cost if res.found else "No solution"}',X,Y+192,'white','body')
            text(f'TIME         {res.elapsed_ms:.2f} ms',X,Y+216,'white','body')
            text('GENERATION & LEARNING',X,Y+257,'white','sub')
            stats=chosen.result.csp.stats
            text(f'CSP BACKTRACKS  {stats.backtracks}',X,Y+287,'white','body')
            text(f'CSP PRUNED      {stats.pruned}',X,Y+311,'white','body')
            text(f'FC {"ON" if enable_fc else "OFF"}   AC-3 {"ON" if enable_ac else "OFF"}',X,Y+335,'gold','body')
            text(f'PLAYER MODEL  {"NN" if controller.use_nn and controller.model else "RULE BASELINE"}',X,Y+359,'cyan','body')
            text(f'ONLINE {controller.adapter.updates}  WAITING {len(controller.adapter.pending)}/10',X,Y+383,'cyan','small')
            text('TAB CLOSES  |  X MEMORY GLITCH',X,Y+405,'muted','small')

        running=True
        while running:
            dt=min(clock.tick(60)/1000,0.05)
            t=monotonic()-game_start;animation_time=t
            for event in pg.event.get():
                if event.type==pg.QUIT:running=False
                if event.type==pg.KEYDOWN:
                    if event.key==pg.K_ESCAPE:running=False
                    elif event.key==pg.K_TAB:ai_lab=not ai_lab;lab_t=0
                    elif event.key==pg.K_r:reset_state()
                    elif event.key==pg.K_x:
                        event_details=controller.adapter.forget_now('manual')
                        if event_details and controller.persist:controller.adapter.save()
                        display_status=('MEMORY GLITCH  //  ADAPTER PARTIALLY ERASED'
                                        if event_details else 'NO ADAPTER MEMORY TO ERASE')
                        status_until=monotonic()+2.5
                    elif event.key==pg.K_v:
                        return 'arena'
                    elif event.key==pg.K_n:
                        controller.use_nn=not controller.use_nn
                        display_status='NEURAL NETWORK '+('ENABLED' if controller.use_nn else 'DISABLED')
                        status_until=monotonic()+2
                    elif event.key==pg.K_f:enable_fc=not enable_fc
                    elif event.key==pg.K_c:enable_ac=not enable_ac
                    elif event.key==pg.K_g:
                        from csp.generator import generate
                        result=generate(seed=controller.seed,target=level.difficulty,
                                        forward_checking=enable_fc,arc_consistency=enable_ac)
                        if result.level:
                            level=result.level;chosen.result=result;reset_state();lab_results={};lab_t=0
                            display_status='NEW CONSTRAINT-VALID LEVEL'
                        else:display_status='NO SOLUTION'
                        status_until=monotonic()+2.5
                    elif event.key in (pg.K_1,pg.K_2,pg.K_3,pg.K_4):
                        lab_algo={pg.K_1:'bfs',pg.K_2:'dfs',pg.K_3:'astar',pg.K_4:'greedy'}[event.key]
                        ai_lab=True;lab_t=0
            keys=pg.key.get_pressed()
            sim_acc+=dt
            while sim_acc>=TICK:
                sim_acc-=TICK
                horizontal=(-1 if keys[pg.K_a] or keys[pg.K_LEFT] else 0)+(
                    1 if keys[pg.K_d] or keys[pg.K_RIGHT] else 0)
                jump=bool(keys[pg.K_SPACE] or keys[pg.K_w] or keys[pg.K_UP])
                act=('JL' if horizontal<0 else 'JR' if horizontal>0 else 'J') if jump else (
                     'L' if horizontal<0 else 'R' if horizontal>0 else 'W')
                old_grounded=level.solid(state.x,state.y) and state.vy==0
                if jump and old_grounded:jumps+=1
                prev=state
                nxt=step(level,state,act)
                if nxt is None:
                    deaths+=1
                    controller.record(level,False,monotonic()-attempt_start,
                                      jumps,landings,state.x/max(level.goal_x,1),
                                      save=True,origin='human')
                    display_status='TRY AGAIN  //  YOUR SKILL PROFILE UPDATED'
                    status_until=monotonic()+2
                    if deaths%3==0:fresh_level()
                    else:reset_state()
                    break
                state=nxt
                grounded=level.solid(state.x,state.y) and state.vy==0
                if grounded and not old_grounded:landings+=1
                # Collect rewards without mutating the level/search representation.
                for row in (state.y-1,state.y-2):
                    if level.tile(state.x,row)=='o' and (state.x,row) not in tokens:
                        tokens.add((state.x,row));score+=10
                if level.complete(state):
                    controller.record(level,True,monotonic()-attempt_start,
                                      jumps,landings,1.0,save=True,origin='human')
                    fresh_level();break
            draw_background(t)
            draw_world(t,min(1,sim_acc/TICK))
            draw_hud(t)
            draw_lab()
            pg.display.flip()
    finally:
        pg.quit()

if __name__=='__main__':run()
