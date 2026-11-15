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

Returns 'manual' (M), 'title' (Esc) or None (window closed).
"""
from __future__ import annotations
import gc
import sys
import threading
from math import sin, floor, log10
from pathlib import Path
from time import perf_counter
from integration.arena import make_arena, STYLE, ALGORITHMS
from game import ui
from game.ui import WIDTH, HEIGHT, PALETTE as P, ALGO_COLORS, ALGO_LABELS

MARGIN, GAP, HEADER, FOOTER = 14, 10, 62, 40
CARD_W = (WIDTH - 2*MARGIN - GAP)//2
CARD_H = (HEIGHT - HEADER - FOOTER - GAP)//2
TILE = 15
STRIP = 34                # coloured title strip at the top of a card
WORLD_X = 8               # world viewport inside a card
WORLD_Y = STRIP + 4
WORLD_W = CARD_W - 2*WORLD_X
WORLD_H = 12*TILE         # enough for all 12 tile rows
STATS_Y = WORLD_Y + WORLD_H + 16
REPLANS_X = 396           # x of the replans / events column
THEMES = ALGO_COLORS
STYLE_SUMMARY = {
 'bfs':'wide frontier, shortest plan',
 'dfs':'deep dives, non-optimal route',
 'astar':'cost + admissible heuristic',
 'greedy':'heuristic only, goal chaser',
}
FRAME_WAIT = 0.004        # seconds a frame may wait for the lanes' ticks
BUSY_BADGE_AFTER = 0.15   # seconds of planning before a lane shows a badge
PULSE_SECONDS = 0.9       # card border flash after a surprise or a finished attempt
FOOTER_KEYS = [('Space', 'Pause'), ('+ / -', 'Speed'), ('H', 'Routes'), ('E', 'Surprise'),
               ('U', 'Surprises on/off'), ('X', 'Glitch'), ('R', 'Restart'), ('M', 'Manual'),
               ('F1', 'Help'), ('Esc', 'Menu')]


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
        screen=ui.open_window(pg,'NEUROFORGE  //  FOUR AGENTS, FOUR EVOLVING WORLDS')
        clock=pg.time.Clock()
        kit=ui.Kit(pg)
        show_paths=True;paused=False;speed=2.0;show_help=False
        elapsed=0;accumulator=0.0
        status='FOUR INDEPENDENT SEARCH AGENTS  //  FOUR PERSONALIZED WORLDS'
        status_until=4.0

        def build_background():
            """Everything that never changes: sky, stars, header and footer."""
            backdrop=ui.Backdrop(pg,seed=seed,planet=(-400,-400))   # no planet behind the header
            bg=pg.Surface((WIDTH,HEIGHT)).convert()
            backdrop.draw(bg,0.0,0.0)
            shade=pg.Surface((WIDTH,HEIGHT),pg.SRCALPHA);shade.fill((6,8,22,110))
            bg.blit(shade,(0,0))
            logo=kit.glow_text(bg,'NEUROFORGE',(MARGIN+4,HEADER//2),P['cyan'],'h1','midleft',strength=130)
            kit.text(bg,'ARENA  //  FOUR SEARCH MINDS',(logo.right+14,HEADER//2+1),P['muted'],'h3','midleft')
            band=pg.Surface((WIDTH,FOOTER),pg.SRCALPHA);band.fill((6,8,22,200))
            bg.blit(band,(0,HEIGHT-FOOTER))
            kit.chips_row(bg,FOOTER_KEYS,HEIGHT-FOOTER+9)
            return bg

        background=build_background()

        def build_view(n):
            """Per-lane render cache: the static card pre-composited over the sky."""
            algo=ALGORITHMS[n];color=THEMES[algo]
            box=(MARGIN+(n%2)*(CARD_W+GAP),HEADER+(n//2)*(CARD_H+GAP))
            card=pg.Surface((CARD_W,CARD_H),pg.SRCALPHA)
            pg.draw.rect(card,(*P['panel'],240),card.get_rect(),border_radius=14)
            pg.draw.rect(card,(*color,38),(0,0,CARD_W,STRIP),
                         border_top_left_radius=14,border_top_right_radius=14)
            pg.draw.rect(card,color,(0,9,4,STRIP-18),border_radius=2)
            title=kit.text(card,STYLE[algo]['title'],(16,STRIP//2),color,'h2','midleft')
            kit.text(card,STYLE_SUMMARY[algo],(title.right+12,STRIP//2+1),P['muted'],'small','midleft')
            badge=ALGO_LABELS[algo]
            badge_w=kit.fonts['key'].size(badge)[0]+20
            kit.pill(card,(CARD_W-14-badge_w//2,STRIP//2),badge,color,fill=color)
            pg.draw.rect(card,(11,14,34),(WORLD_X,WORLD_Y,WORLD_W,WORLD_H),border_radius=8)
            pg.draw.rect(card,P['panel_hi'],(WORLD_X,WORLD_Y+WORLD_H+5,WORLD_W,4),border_radius=2)
            for x,label in ((14,'WINS / LOSSES'),(128,'NN P(WIN)'),(232,'NODES EXPANDED'),
                            (REPLANS_X,'REPLANS / EVENTS')):
                kit.text(card,label,(x,STATS_Y+22),P['dim'],'tiny')
            base=pg.Surface((CARD_W,CARD_H)).convert()
            base.blit(background,(0,0),(box[0],box[1],CARD_W,CARD_H))
            base.blit(card,(0,0))
            panel=base.copy()
            kit.text(panel,'GENERATING WORLD ...',(CARD_W//2,WORLD_Y+WORLD_H//2),P['muted'],'h2','center')
            return {'box':box,'color':color,'base':base,'panel':panel,
                    'level':None,'tiles':None,'coins':(),
                    'seen':None,'pulse_until':0.0,'pulse_color':color}

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
                        pg.draw.rect(layer,P['tile'],(sx,posy,TILE,TILE))
                        if not level.solid(xx,yy-1):
                            pg.draw.rect(layer,P['tile_top'],(sx,posy,TILE,4),border_radius=1)
                        if (xx+yy)%3==0:
                            pg.draw.line(layer,P['tile_line'],(sx+3,posy+TILE-4),(sx+TILE-4,posy+TILE-4))
                    elif tile=='^':
                        for part in range(2):
                            points=[(sx+part*TILE//2+1,posy+TILE),
                                    (sx+part*TILE//2+TILE//4,posy+3),
                                    (sx+(part+1)*TILE//2,posy+TILE)]
                            pg.draw.polygon(layer,P['spike'],points)
                    elif tile=='o':
                        coins.append((xx,yy))
            return layer,tuple(coins)

        def draw_panel(view,agent,t,alpha):
            algo=agent.algorithm;color=view['color']
            p=view['panel']
            p.blit(view['base'],(0,0))
            level=agent.level
            if view['level'] is not level:
                view['level']=level
                view['tiles'],view['coins']=build_tiles(level)
            seen=(agent.total_surprises,agent.wins,agent.losses)
            if view['seen'] is not None and seen!=view['seen']:
                old=view['seen']
                view['pulse_until']=t+PULSE_SECONDS
                view['pulse_color']=(P['pink'] if seen[0]!=old[0] else
                                     P['green'] if seen[1]!=old[1] else P['red'])
            view['seen']=seen
            world_label=f'WORLD {agent.stages:02}'
            badge_w=kit.fonts['key'].size(ALGO_LABELS[algo])[0]+20
            kit.text(p,world_label,(CARD_W-24-badge_w,STRIP//2),P['muted'],'tiny','midright')
            wy=WORLD_Y
            full_w=level.width*TILE
            # Sprites glide from the previous logical state to the current one.
            state=agent.state;prev=agent.prev
            fx,fy=ui.lerp_cell((prev.x,prev.y),(state.x,state.y),alpha)
            camera=max(0,min(max(0,full_w-WORLD_W),fx*TILE-WORLD_W*0.35))
            ox=floor(WORLD_X-camera)   # one shared integer scroll offset: no tile/sprite shimmer
            p.set_clip(pg.Rect(WORLD_X,WORLD_Y,WORLD_W,WORLD_H))
            # chunk boundaries: each world is a sequence of CSP-chosen chunks
            chunks=level.chunks
            for k in range(len(chunks)+1):
                xx=ox+k*8*TILE
                if WORLD_X<=xx<CARD_W-WORLD_X:
                    pg.draw.line(p,(30,40,72),(xx,wy+1),(xx,wy+WORLD_H-1))
                    if k<len(chunks):
                        kit.text(p,chunks[k][:12].upper(),(xx+5,wy+4),(70,86,124),'tiny')
            p.blit(view['tiles'],(ox,wy))
            collected=agent.coins
            for xx,yy in view['coins']:
                sx=ox+xx*TILE
                if -TILE<=sx<=CARD_W and (xx,yy) not in collected:
                    py=wy+yy*TILE+TILE//2+int(sin(t*5+xx)*2)
                    pg.draw.circle(p,P['gold'],(sx+TILE//2,py),4,2)
            # The world can change mid-run. Draw the actual phase-dependent
            # obstacles used by physics, not an unrelated visual animation.
            phase=state.phase;before=prev.phase
            for platform in level.platforms:
                mx,my=ui.lerp_cell(platform.position(before),platform.position(phase),alpha)
                px=ox+int(mx*TILE);py=int(wy+my*TILE)
                if -platform.width*TILE<px<CARD_W:
                    pg.draw.rect(p,(89,194,217),(px,py,platform.width*TILE,TILE//2),border_radius=4)
                    pg.draw.rect(p,(201,246,249),(px+2,py+1,platform.width*TILE-4,2),border_radius=1)
            for enemy in level.enemies:
                mx,my=ui.lerp_cell(enemy.position(before),enemy.position(phase),alpha)
                px=ox+int(mx*TILE)+TILE//2;py=int(wy+my*TILE)+TILE//2
                if -TILE<px<CARD_W+TILE:
                    pg.draw.circle(p,(239,94,130),(px,py),TILE//2)
                    pg.draw.circle(p,(253,226,220),(px-2,py-2),3)
                    pg.draw.circle(p,(42,37,78),(px-1,py-2),1)
            for gate in level.gates:
                gx=ox+gate.x*TILE+TILE//2
                if -TILE<gx<CARD_W+TILE:
                    active=gate.active(phase)
                    top=wy+gate.y_top*TILE;bottom=wy+(gate.y_bottom+1)*TILE
                    if active:
                        pg.draw.line(p,(120,30,80),(gx,top),(gx,bottom),7)
                        pg.draw.line(p,(255,120,180),(gx,top),(gx,bottom),3)
                    else:
                        for yy in range(top,bottom,8):
                            pg.draw.line(p,(78,102,137),(gx,yy),(gx,min(bottom,yy+4)),2)
            if show_paths and agent.plan.found:
                path=agent.plan.path[max(0,agent.plan_index):]
                if path:
                    half=TILE//2;cx=ox+half;cy=wy-TILE+half
                    coords=[(cx+s.x*TILE,cy+s.y*TILE) for s in path]
                    if len(coords)>1:
                        pg.draw.lines(p,(10,12,30),False,coords,4)
                        pg.draw.lines(p,color,False,coords,2)
            gx=ox+level.goal_x*TILE
            if 0<gx<CARD_W:
                py=wy+(level.height-2)*TILE-22
                r=3*sin(t*4)
                pg.draw.ellipse(p,P['cyan'],(gx-2-r/2,py-18-r,22+r,40+2*r),2)
                pg.draw.ellipse(p,(239,182,255),(gx+4,py-8,10,20),2)
            # individual sprite; color changes with search method
            px=ox+int(fx*TILE)+TILE//2
            py=int(wy+(fy-1)*TILE)+TILE//2
            pg.draw.circle(p,ui.mix(P['panel'],color,0.35),(px,py),13)
            pg.draw.rect(p,(20,31,58),(px-6,py-8,12,16),border_radius=4)
            pg.draw.rect(p,color,(px-6,py-8,12,16),width=2,border_radius=4)
            pg.draw.rect(p,(227,251,254),(px-4,py-4,9,5),border_radius=2)
            p.set_clip(None)
            # progress along the level (the track is part of the card)
            frac=fx/max(1,level.goal_x)
            pg.draw.rect(p,color,(WORLD_X,WORLD_Y+WORLD_H+5,max(4,int(WORLD_W*min(1,frac))),4),border_radius=2)
            # statistics
            y=STATS_Y
            kit.text(p,f'{agent.wins:02} / {agent.losses:02}',(14,y),P['text'],'numsm')
            kit.text(p,f'{agent.choice.predicted_win:.0%}',(128,y),color,'numsm')
            expanded=agent.plan.expanded
            value=kit.text(p,f'{expanded:,}',(232,y),P['text'],'numsm')
            kit.bar(p,(value.right+8,y+8,max(30,REPLANS_X-24-value.right-8),6),
                    log10(expanded+1)/log10(24001),ui.mix(P['panel_hi'],color,0.8))
            kit.text(p,f'{agent.replans} / {agent.surprise_count}',(REPLANS_X,y),P['text'],'numsm')
            kit.text(p,f'{level.difficulty:.2f}',(CARD_W-14,y),ui.mix(P['green'],P['pink'],level.difficulty),
                     'numsm','topright')
            kit.text(p,f'DIFF / TARGET {agent.controller.challenge:.2f}',(CARD_W-14,y+22),P['dim'],
                     'tiny','topright')
            ad=agent.controller.adapter
            to_forget=(ad.forget_every-(ad.updates%ad.forget_every))*ad.batch_size-len(ad.pending)
            outcome=agent.last_event
            if ad.recovery_updates_remaining:
                outcome=(f'RE-LEARNING / BOOST x{ad.recovery_boost:g} / '
                         f'{ad.recovery_updates_remaining} FRESH BATCH(ES) LEFT')
            recent=ad.last_forget_event
            if recent and recent['observations']==ad.observations:
                outcome=f"MEMORY GLITCH: {recent['erased_count']} COEFFICIENTS ERASED"
            hot=('MEMORY' in outcome or 'RE-LEARNING' in outcome)
            surprise=outcome.startswith('SURPRISE')
            online=(f'ONLINE {ad.updates}  |  NEW {len(ad.pending)}/10  |  '
                    + (f'DROP IN {to_forget:02}' if ad.forgetting_enabled else 'GLITCHES OFF'))
            right=kit.text(p,online,(CARD_W-14,CARD_H-14),P['gold'] if ad.forgetting_enabled else P['dim'],
                           'tiny','midright')
            kit.text(p,outcome,(14,CARD_H-14),P['gold'] if hot else P['pink'] if surprise else P['muted'],
                     'small','midleft',width=right.left-34)

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

        def draw_header():
            if elapsed<status_until:
                fade=255*min(1.0,(status_until-elapsed)*2)
                kit.text(screen,status,(WIDTH//2+40,HEADER//2),P['gold'],'h3','center',alpha=fade,width=470)
            x=WIDTH-MARGIN
            for label,color in ((('PAUSED' if paused else 'LIVE'),P['pink'] if paused else P['green']),
                                (f'{speed:g}x',P['text']),
                                ('SURPRISES '+('ON' if surprises_on else 'OFF'),P['pink'] if surprises_on else P['dim']),
                                ('GLITCHES '+('ON' if forgetting_on else 'OFF'),P['gold'] if forgetting_on else P['dim'])):
                w=kit.fonts['key'].size(label)[0]+20
                kit.pill(screen,(x-w//2,HEADER//2),label,color,fill=color if label in ('LIVE','PAUSED') else None)
                x-=w+8

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
                for view in views:view['seen']=None
                forgetting_on=lanes[0].agent.controller.adapter.forgetting_enabled
                surprises_on=lanes[0].agent.surprises
                if build_status:
                    status=build_status;status_until=elapsed+3
                build=None
            for event in pg.event.get():
                if event.type==pg.QUIT:running=False
                elif event.type==pg.KEYDOWN:
                    if show_help:
                        if event.key in (pg.K_ESCAPE,pg.K_F1):show_help=False
                        continue
                    if event.key==pg.K_ESCAPE:return 'title'
                    elif event.key==pg.K_m:return 'manual'
                    elif event.key==pg.K_F1:show_help=True
                    elif event.key==pg.K_F11:ui.toggle_fullscreen(pg)
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
            if lanes and not paused and not show_help:
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
            draw_header()
            for n,view in enumerate(views):
                lane=lanes[n] if n<len(lanes) else None
                if lane is not None and lane.idle.is_set():
                    draw_panel(view,lane.agent,elapsed,alpha)
                screen.blit(view['panel'],view['box'])
                x0,y0=view['box']
                card=pg.Rect(x0,y0,CARD_W,CARD_H)
                left=view['pulse_until']-elapsed
                if left>0:
                    pg.draw.rect(screen,ui.mix(P['line'],view['pulse_color'],left/PULSE_SECONDS),card,
                                 width=2,border_radius=14)
                else:
                    pg.draw.rect(screen,P['line'],card,width=1,border_radius=14)
                if lane is not None and not lane.idle.is_set() and now-lane.busy_since>BUSY_BADGE_AFTER:
                    kit.pill(screen,(x0+CARD_W//2,y0+WORLD_Y+16),'PLANNING ...',view['color'])
            if paused and not show_help:
                kit.pill(screen,(WIDTH//2,HEADER+CARD_H+GAP//2),'PAUSED  //  SPACE TO RESUME',P['pink'],
                         face='h3',height=30,pad=16)
            if show_help:
                kit.overlay_help(screen,'CONTROLS  //  ARENA',ui.ARENA_HELP)
            pg.display.flip()
    finally:
        for lane in lanes:lane.stop()
        for lane in lanes:lane.join(0.25)
        sys.setswitchinterval(switch_interval)
        gc.unfreeze()
        if gc_was_enabled:gc.enable()
        pg.quit()

if __name__=='__main__':run_arena()
