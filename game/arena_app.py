"""NEUROFORGE ARENA: four autonomous live platformer lanes in one window.

The agents' *search algorithms* select every action. A frozen trained MLP plus
an independent online calibration adapter coaches each lane's procedural CSP
level selection. Each lane runs its own physics simulation and search plans.
Their differences are real graph-search behavior + explicitly described execution
noise, NOT four separately named copies of the same hidden pathfinder.

Smoothness: the window never waits for planning. Each lane's simulation runs on
its own worker thread; a lane that is busy replanning or generating its next
world keeps its last drawn picture while the other lanes and the window keep
animating. Static artwork (background, cards, level tiles, text) is rendered
once and reused, and sprites are interpolated between logical ticks. None of
this changes what an agent does: every lane still executes exactly the same
sequence of ticks as the headless engine.
"""
from __future__ import annotations
import gc
import sys
import threading
from math import sin, floor
from pathlib import Path
from random import Random
from time import perf_counter
from integration.arena import make_arena, STYLE, ALGORITHMS

WIDTH, HEIGHT = 1372, 926
MARGIN, GAP, HEADER, FOOTER = 18, 14, 100, 56
CARD_W = (WIDTH - 2*MARGIN - GAP)//2
CARD_H = (HEIGHT - HEADER - FOOTER - GAP - MARGIN)//2
TILE = 17
WORLD_Y = 65              # top of the world viewport inside a card
WORLD_H = 12*TILE         # enough for all 12 tile rows
SKY = (10,13,30)
WHITE = (231,241,251)
MUTED = (146,158,188)
THEMES = {
 'bfs':(89,218,236), 'dfs':(181,121,248),
 'astar':(111,237,170), 'greedy':(255,117,164)}
STYLE_SUMMARY = {
 'bfs':'WIDE FRONTIER / SHORTEST ACTION PLAN',
 'dfs':'DEEP EXPLORATION / NONOPTIMAL ROUTE',
 'astar':'COST + ADMISSIBLE HEURISTIC',
 'greedy':'HEURISTIC ONLY / GOAL CHASER',
}
FRAME_WAIT = 0.004        # seconds a frame may wait for the lanes' ticks
BUSY_BADGE_AFTER = 0.15   # seconds of planning before a lane shows a badge


class Lane:
    """Runs one agent's simulation on a worker thread, off the render thread.

    The render thread touches `agent` only while `idle` is set, and the worker
    only between submit() and idle.set(), so no further locking is needed.
    Ticks are executed strictly in order; a busy lane simply receives no new
    ticks until it has finished thinking.
    """
    def __init__(self, agent):
        self.agent = agent
        self.idle = threading.Event(); self.idle.set()
        self.error = None
        self.deferred = []            # (action(agent), results list) run before the next ticks
        self.busy_since = 0.0
        self._wake = threading.Event()
        self._job = None
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='lane-'+agent.algorithm)
        self._thread.start()

    def _run(self):
        while True:
            self._wake.wait(); self._wake.clear()
            if not self._alive: return
            job, self._job = self._job, None
            try:
                if job is not None: job()
            except BaseException as ex:   # re-raised on the render thread
                self.error = ex
            finally:
                self.idle.set()

    def submit(self, ticks):
        """Queue deferred key actions, then `ticks` logical steps. Idle lanes only."""
        actions, self.deferred = self.deferred, []
        agent = self.agent
        def job():
            for action, results in actions:
                results.append(action(agent))
            for _ in range(ticks):
                if not self._alive: break
                agent.tick()
        self.busy_since = perf_counter()
        self.idle.clear()
        self._job = job
        self._wake.set()

    def stop(self):
        self._alive = False
        self._wake.set()

    def join(self, timeout):
        self._thread.join(timeout)


def run_arena(seed=42, persist=True, forget_every=3, forget_fraction=.35,
              feature_dropout=.12):
    try:
        import pygame as pg
    except ImportError as ex:
        raise RuntimeError('Install Pygame CE: python -m pip install pygame-ce==2.5.8') from ex
    pg.init()
    lanes=[]
    switch_interval=sys.getswitchinterval()
    # Let the render thread get the interpreter back quickly while a lane plans.
    sys.setswitchinterval(0.001)
    # The planners allocate hundreds of thousands of objects but create no
    # reference cycles (measured: zero cyclic garbage), so automatic cycle
    # collection is pure stop-the-world pauses here (measured up to ~130 ms).
    # It is suspended while the arena runs; a full pass still happens whenever
    # the arena is rebuilt (R) and normal collection resumes on exit.
    gc_was_enabled=gc.isenabled()
    gc.collect();gc.freeze();gc.disable()
    try:
        screen=pg.display.set_mode((WIDTH,HEIGHT))
        pg.display.set_caption('NEUROFORGE  //  FOUR AGENTS, FOUR EVOLVING WORLDS')
        clock=pg.time.Clock()
        fonts={
            'logo':pg.font.SysFont('dejavusans',31,bold=True),
            'heading':pg.font.SysFont('dejavusans',20,bold=True),
            'sub':pg.font.SysFont('dejavusans',15,bold=True),
            'body':pg.font.SysFont('dejavusans',13),
            'small':pg.font.SysFont('dejavusans',11),
            'tiny':pg.font.SysFont('dejavusans',10),
        }
        show_paths=True;paused=False;speed=2.0
        elapsed=0;accumulator=0.0
        rng=Random(seed)
        stars=[(rng.random()*WIDTH,rng.random()*120,1+rng.random()*3) for _ in range(120)]
        status='FOUR INDEPENDENT SEARCH AGENTS  //  FOUR PERSONALIZED WORLDS'
        status_until=0.0
        text_cache={}

        def text_surface(s, color, face, limit):
            """Rendered text is reused until the string itself changes."""
            key=(s,color,face,limit)
            out=text_cache.get(key)
            if out is None:
                if len(text_cache)>4000:text_cache.clear()
                font=fonts[face]
                out=font.render(s,True,color)
                if limit is not None and out.get_width()>limit:
                    string=s
                    while string and font.size(string+'...')[0]>limit:
                        string=string[:-1]
                    out=font.render(string+'...',True,color)
                text_cache[key]=out
            return out

        def txt(s, pos, color=WHITE, face='body', anchor='left', limit=None, surface=None):
            dest=screen if surface is None else surface
            out=text_surface(str(s),color,face,limit)
            x,y=pos
            if anchor=='center':x-=out.get_width()//2
            elif anchor=='right':x-=out.get_width()
            dest.blit(out,(int(x),int(y)))

        def rr(surface,color,rect,radius=10,border=0):
            pg.draw.rect(surface,color,rect,width=border,border_radius=radius)

        def build_background():
            """Everything that never changes: sky, stars, header and footer."""
            bg=pg.Surface((WIDTH,HEIGHT)).convert()
            bg.fill(SKY)
            # calm star field + identity stripes
            for x,y,r in stars:
                pg.draw.circle(bg,(35,49,84),(int(x),int(y)),int(r))
            pg.draw.line(bg,(61,83,124),(MARGIN,89),(WIDTH-MARGIN,89),1)
            txt('NEUROFORGE', (MARGIN+5,17),(119,226,239),'logo',surface=bg)
            txt('THE FOUR MINDS EXPERIMENT', (MARGIN+6,61),MUTED,'sub',surface=bg)
            txt('BFS  /  DFS  /  A*  /  GREEDY', (WIDTH-28,20),WHITE,'heading','right',surface=bg)
            txt('World surprises invalidate routes: agents observe changes and replan live.',
                (WIDTH-29,60),MUTED,'body','right',surface=bg)
            rr(bg,(18,23,46),(MARGIN,HEIGHT-FOOTER+6,WIDTH-2*MARGIN,FOOTER-14),10)
            txt('SPACE  PAUSE    +/- SPEED    H  ROUTES    U  RANDOM WORLD ON/OFF    E  SURPRISE NOW',
                (MARGIN+14,HEIGHT-FOOTER+14),WHITE,'body',surface=bg)
            txt('D  GLITCH ON/OFF    X  GLITCH NOW    M  MANUAL    R  RESTART    S  SAVE    ESC  EXIT',
                (MARGIN+14,HEIGHT-FOOTER+34),MUTED,'small',surface=bg)
            return bg

        background=build_background()

        def build_view(n):
            """Per-lane render cache: the static card pre-composited over the sky."""
            algo=ALGORITHMS[n];color=THEMES[algo]
            box=(MARGIN+(n%2)*(CARD_W+GAP),HEADER+(n//2)*(CARD_H+GAP))
            card=pg.Surface((CARD_W,CARD_H),pg.SRCALPHA)
            rr(card,(20,25,49),card.get_rect(),14)
            rr(card,(69,79,111),card.get_rect().inflate(-1,-1),13,1)
            rr(card,(*color,25),(0,0,CARD_W,73),13)
            pg.draw.rect(card,color,(0,0,5,CARD_H),border_radius=2)
            txt(STYLE[algo]['title'],(16,9),color,'heading',surface=card)
            txt(STYLE_SUMMARY[algo],(16,41),MUTED,'small',surface=card)
            txt(algo.upper(),(CARD_W-14,12),WHITE,'sub','right',surface=card)
            pg.draw.rect(card,(16,23,46),(7,WORLD_Y,CARD_W-14,WORLD_H),border_radius=6)
            rr(card,(32,40,65),(12,WORLD_Y+WORLD_H+9,CARD_W-24,6),3)
            base=pg.Surface((CARD_W,CARD_H)).convert()
            base.blit(background,(0,0),(box[0],box[1],CARD_W,CARD_H))
            base.blit(card,(0,0))
            panel=base.copy()
            txt('GENERATING WORLD ...',(CARD_W//2,WORLD_Y+WORLD_H//2-8),MUTED,'sub','center',surface=panel)
            return {'box':box,'color':color,'base':base,'panel':panel,
                    'level':None,'tiles':None,'coins':()}

        views=[build_view(n) for n in range(len(ALGORITHMS))]

        def build_tiles(level):
            """Static geometry of one Level, drawn once and scrolled as a single blit."""
            layer=pg.Surface((level.width*TILE,level.height*TILE),pg.SRCALPHA)
            coins=[]
            for xx in range(level.width):
                sx=xx*TILE
                for yy in range(level.height):
                    tile=level.tile(xx,yy)
                    posy=yy*TILE
                    if tile=='#':
                        rr(layer,(46,49,80),(sx,posy,TILE,TILE),2)
                        if not level.solid(xx,yy-1):
                            rr(layer,(60,157,153),(sx,posy,TILE,4),1)
                        if (xx+yy)%3==0:
                            pg.draw.line(layer,(65,66,98),(sx+3,posy+TILE-5),(sx+TILE-4,posy+TILE-5))
                    elif tile=='^':
                        for part in range(2):
                            points=[(sx+part*TILE//2+1,posy+TILE),
                                    (sx+part*TILE//2+TILE//4,posy+3),
                                    (sx+(part+1)*TILE//2,posy+TILE)]
                            pg.draw.polygon(layer,(239,86,127),points)
                    elif tile=='o':
                        coins.append((xx,yy))
            return layer,tuple(coins)

        def lerp_cell(a,b,alpha):
            """Blend two grid positions; snap when they are not one continuous move."""
            if abs(b[0]-a[0])>2 or abs(b[1]-a[1])>4:return b
            return a[0]+(b[0]-a[0])*alpha,a[1]+(b[1]-a[1])*alpha

        def draw_panel(view,agent,t,alpha):
            algo=agent.algorithm;color=view['color']
            p=view['panel']
            p.blit(view['base'],(0,0))
            level=agent.level
            if view['level'] is not level:
                view['level']=level
                view['tiles'],view['coins']=build_tiles(level)
            txt(f'WORLD  {agent.stages:02}',(CARD_W-14,41),MUTED,'small','right',surface=p)
            wy=WORLD_Y
            world_height=WORLD_H
            view_w=CARD_W-24
            full_w=level.width*TILE
            # Sprites glide from the previous logical state to the current one.
            state=agent.state;prev=agent.prev
            fx,fy=lerp_cell((prev.x,prev.y),(state.x,state.y),alpha)
            camera=max(0,min(max(0,full_w-view_w),fx*TILE-view_w*0.35))
            ox=floor(12-camera)   # one shared integer scroll offset: no tile/sprite shimmer
            for yy in range(4):
                offset=(yy*17+camera*0.11*(yy+1))%(view_w+210)
                pg.draw.circle(p,(26+yy*4,43+yy*4,70+yy*2),
                               (int((view_w-offset)%(view_w+100)),wy+25+yy*43),35+yy*13,2)
            # vertical grid accent reinforces procedural chunks and independent levels
            chunks=level.chunks
            for k in range(len(chunks)+1):
                xx=ox+k*8*TILE
                if 8<=xx<CARD_W-8:
                    pg.draw.line(p,(36,55,80),(xx,wy+1),(xx,wy+world_height-1))
                    if k<len(chunks):
                        txt(chunks[k][:10].upper(),(xx+5,wy+4),(88,110,139),'tiny',surface=p)
            p.blit(view['tiles'],(ox,wy))
            collected=agent.coins
            for xx,yy in view['coins']:
                sx=ox+xx*TILE
                if -TILE<=sx<=CARD_W and (xx,yy) not in collected:
                    py=wy+yy*TILE+TILE//2+int(sin(t*5+xx)*2)
                    pg.draw.circle(p,(250,212,124),(sx+TILE//2,py),4,2)
            # The world can change mid-run. Draw the actual phase-dependent
            # obstacles used by physics, not an unrelated visual animation.
            phase=state.phase;before=prev.phase
            for platform in level.platforms:
                mx,my=lerp_cell(platform.position(before),platform.position(phase),alpha)
                px=ox+int(mx*TILE);py=int(wy+my*TILE)
                if -platform.width*TILE<px<CARD_W:
                    rr(p,(89,194,217),(px,py,platform.width*TILE,TILE//2),4)
                    rr(p,(201,246,249),(px+2,py+1,platform.width*TILE-4,3),2)
                    for bolt in range(platform.width):
                        pg.draw.circle(p,(27,78,108),(px+8+bolt*TILE,py+7),2)
            for enemy in level.enemies:
                mx,my=lerp_cell(enemy.position(before),enemy.position(phase),alpha)
                px=ox+int(mx*TILE)+TILE//2;py=int(wy+my*TILE)+TILE//2
                if -TILE<px<CARD_W+TILE:
                    pg.draw.circle(p,(239,94,130),(px,py),TILE//2)
                    pg.draw.circle(p,(253,202,194),(px-2,py-2),3)
                    pg.draw.circle(p,(42,37,78),(px-1,py-2),1)
                    pg.draw.line(p,(255,163,136),(px-5,py+5),(px+5,py+5),2)
            for gate in level.gates:
                gx=ox+gate.x*TILE+TILE//2
                if -TILE<gx<CARD_W+TILE:
                    active=gate.active(phase)
                    gate_color=(255,80,151) if active else (78,102,137)
                    pg.draw.line(p,gate_color,(gx,wy+gate.y_top*TILE),
                                 (gx,wy+(gate.y_bottom+1)*TILE),4 if active else 2)
                    pg.draw.circle(p,(255,185,198) if active else (89,108,143),
                                   (gx,wy+gate.y_top*TILE),4)
            if show_paths and agent.plan.found:
                path=agent.plan.path[max(0,agent.plan_index):]
                if path:
                    half=TILE//2;cx=ox+half;cy=wy-TILE+half
                    coords=[(cx+s.x*TILE,cy+s.y*TILE) for s in path]
                    if len(coords)>1:pg.draw.lines(p,color,False,coords,2)
                    for px,py in coords[::4]:
                        if 5<=px<=CARD_W-5:
                            pg.draw.circle(p,color,(px,py),2)
            gx=ox+level.goal_x*TILE
            if 0<gx<CARD_W:
                py=wy+9*TILE-13
                pg.draw.ellipse(p,(108,221,201),(gx,py-19,24,51),3)
                pg.draw.circle(p,(230,230,255),(gx+12,py+7),8,2)
            # individual sprite; color changes with search method
            px=ox+int(fx*TILE)+TILE//2
            py=int(wy+(fy-1)*TILE)+TILE//2
            pg.draw.circle(p,color,(px,py),14,2)
            rr(p,(20,31,58),(px-7,py-8,14,18),5)
            rr(p,color,(px-7,py-8,14,18),5,2)
            rr(p,(227,251,254),(px-5,py-4,10,6),2)
            pg.draw.circle(p,(23,36,71),(px-1,py-1),1)
            pg.draw.circle(p,(23,36,71),(px+3,py-1),1)
            # player progress line / difficulty bar (the track is part of the card)
            info_y=wy+world_height+9
            rr(p,color,(12,info_y,max(2,int((CARD_W-24)*fx/max(1,level.goal_x))),6),3)
            stats_y=info_y+16
            txt(f'W {agent.wins:02}   L {agent.losses:02}',(14,stats_y),WHITE,'sub',surface=p)
            txt(f'DIFF {level.difficulty:.2f}   TARGET {agent.controller.challenge:.2f}',
                (CARD_W//2+5,stats_y),color,'body',surface=p)
            txt(f'EXP {agent.plan.expanded:04}   REPLANS {agent.replans:02}   EVENTS {agent.surprise_count}',(14,stats_y+22),MUTED,'small',surface=p)
            txt(f'NN P(WIN) {agent.choice.predicted_win:.0%}',(CARD_W//2+5,stats_y+22),WHITE,'small',surface=p)
            ad=agent.controller.adapter
            txt(f'ONLINE {ad.updates}  |  NEW {len(ad.pending)}/10',
                (14,stats_y+41),color,'body',surface=p)
            to_forget=(ad.forget_every-(ad.updates%ad.forget_every))*ad.batch_size-len(ad.pending)
            txt(f'MEMORY DROP IN {to_forget:02} ATTEMPTS',
                (CARD_W//2+5,stats_y+41),(255,181,116) if ad.forgetting_enabled else MUTED,'small',surface=p)
            outcome=agent.last_event
            if ad.recovery_updates_remaining:
                outcome=(f'RE-LEARNING / BOOST x{ad.recovery_boost:g} / '
                         f'{ad.recovery_updates_remaining} FRESH BATCH(ES) LEFT')
            recent=ad.last_forget_event
            if recent and recent['observations']==ad.observations:
                outcome=f"MEMORY GLITCH: {recent['erased_count']} COEFFICIENTS ERASED"
            txt(outcome,(14,CARD_H-18),(255,192,123) if ('MEMORY' in outcome or 'RE-LEARNING' in outcome) else MUTED,
                'small',limit=CARD_W-28,surface=p)

        def start_build(build_seed,surprises):
            """Create four agents (levels + first plans) without freezing the window."""
            box={'lanes':None,'error':None}
            def work():
                try:
                    box['lanes']=make_arena(build_seed,persist=persist,forget_every=forget_every,
                                            forget_fraction=forget_fraction,
                                            feature_dropout=feature_dropout,surprises=surprises)
                except BaseException as ex:
                    box['error']=ex
            threading.Thread(target=work,daemon=True,name='arena-build').start()
            return box

        def broadcast(action,report=None):
            """Apply a key action to every lane as soon as that lane is between ticks."""
            results=[]
            for lane in lanes:lane.deferred.append((action,results))
            if report is not None:reports.append((results,len(lanes),report))

        build=start_build(seed,True)
        build_status=None
        reports=[]                 # (results, expected count, message builder)
        surprises_on=True;forgetting_on=True

        running=True
        while running:
            dt=min(clock.tick(60)/1000,0.07)
            elapsed+=dt
            if build is not None and (build['lanes'] is not None or build['error'] is not None):
                if build['error'] is not None:raise build['error']
                for lane in lanes:lane.stop()
                if lanes:gc.collect()   # restart: the one moment a pause is invisible
                lanes=[Lane(agent) for agent in build['lanes']]
                reports.clear();accumulator=0.0
                forgetting_on=lanes[0].agent.controller.adapter.forgetting_enabled
                surprises_on=lanes[0].agent.surprises
                if build_status:
                    status=build_status;status_until=elapsed+3
                build=None
            for event in pg.event.get():
                if event.type==pg.QUIT:running=False
                elif event.type==pg.KEYDOWN:
                    if event.key==pg.K_ESCAPE:running=False
                    elif event.key==pg.K_m:return 'manual'
                    elif event.key==pg.K_SPACE:paused=not paused
                    elif event.key in (pg.K_EQUALS,pg.K_PLUS,pg.K_KP_PLUS,pg.K_RIGHTBRACKET):
                        speed=min(32.0,speed*2)
                    elif event.key in (pg.K_MINUS,pg.K_KP_MINUS,pg.K_LEFTBRACKET):
                        speed=max(.25,speed/2)
                    elif event.key==pg.K_h:show_paths=not show_paths
                    elif event.key==pg.K_s:
                        folder=Path(__file__).resolve().parents[1]/'results'
                        folder.mkdir(parents=True,exist_ok=True)
                        dest=folder/'arena_screenshot.png';pg.image.save(screen,str(dest))
                        status='SCREENSHOT SAVED TO results/arena_screenshot.png'
                        status_until=elapsed+3
                    elif not lanes:pass    # the keys below act on agents that are still booting
                    elif event.key==pg.K_u:
                        surprises_on=not surprises_on
                        for lane in lanes:lane.agent.surprises=surprises_on
                        status='LIVE WORLD SURPRISES '+('ENABLED' if surprises_on else 'DISABLED')
                        status_until=elapsed+2.5
                    elif event.key==pg.K_e:
                        broadcast(lambda a:a.inject_surprise(),
                                  lambda r:f'WORLD SHIFT  //  {sum(r)}/4 AGENTS OBSERVED AND REPLANNED')
                        status='WORLD SHIFT  //  AGENTS ARE OBSERVING AND REPLANNING'
                        status_until=elapsed+3
                    elif event.key==pg.K_d:
                        forgetting_on=not forgetting_on
                        def set_forgetting(a,enabled=forgetting_on):
                            a.controller.adapter.forgetting_enabled=enabled
                            if a.persist:a.controller.adapter.save()
                        broadcast(set_forgetting)
                        status='EPISODIC MEMORY GLITCHES '+('ENABLED' if forgetting_on else 'DISABLED')
                        status_until=elapsed+2.5
                    elif event.key==pg.K_x:
                        def glitch(a):
                            ev=a.controller.adapter.forget_now('player-triggered')
                            if not ev:return False
                            a.last_event='MANUAL MEMORY GLITCH'
                            if a.persist:a.controller.adapter.save()
                            return True
                        broadcast(glitch,
                                  lambda r:f'MANUAL MEMORY GLITCH  //  {sum(r)}/4 ADAPTERS PARTIALLY ERASED')
                    elif event.key==pg.K_r and build is None:
                        build=start_build(seed+int(elapsed),surprises_on)
                        build_status='FOUR AGENTS RESTARTED'
                        status='GENERATING FOUR NEW WORLDS ...';status_until=elapsed+30
            ticks=0
            if lanes and not paused:
                accumulator+=dt*10*speed
                while accumulator>=1 and ticks<42:
                    accumulator-=1
                    ticks+=1
            started=[]
            for lane in lanes:
                if lane.error is not None:raise lane.error
                if lane.idle.is_set() and (ticks or lane.deferred):
                    lane.submit(ticks);started.append(lane)
            # Ordinary ticks finish in well under a millisecond, so normally every
            # lane is drawn fully up to date. A lane that turns out to be replanning
            # or building its next world is not waited for beyond this small budget,
            # and lanes already known to be busy are not waited for at all.
            deadline=perf_counter()+FRAME_WAIT
            for lane in started:
                if not lane.idle.is_set():
                    lane.idle.wait(max(0.0,deadline-perf_counter()))
            for item in reports[:]:
                results,expected,report=item
                if len(results)>=expected:
                    reports.remove(item)
                    status=report(results);status_until=elapsed+3
            alpha=min(1.0,accumulator)
            now=perf_counter()
            screen.blit(background,(0,0))
            if elapsed<status_until:
                txt(status,(WIDTH//2,76),(255,205,122),'small','center')
            for n,view in enumerate(views):
                lane=lanes[n] if n<len(lanes) else None
                if lane is not None and lane.idle.is_set():
                    draw_panel(view,lane.agent,elapsed,alpha)
                screen.blit(view['panel'],view['box'])
                if lane is not None and not lane.idle.is_set() and now-lane.busy_since>BUSY_BADGE_AFTER:
                    x0,y0=view['box']
                    rr(screen,(20,25,49),(x0+CARD_W//2-52,y0+WORLD_Y+6,104,20),8)
                    txt('PLANNING ...',(x0+CARD_W//2,y0+WORLD_Y+9),view['color'],'small','center')
            txt(f'{speed:g}x  /  {"PAUSED" if paused else "LIVE"}',
                (WIDTH-MARGIN-13,HEIGHT-FOOTER+23),THEMES['astar'] if not paused else THEMES['greedy'],
                'sub','right')
            pg.display.flip()
    finally:
        for lane in lanes:lane.stop()
        for lane in lanes:lane.join(0.25)
        sys.setswitchinterval(switch_interval)
        gc.unfreeze()
        if gc_was_enabled:gc.enable()
        pg.quit()

if __name__=='__main__':run_arena()
