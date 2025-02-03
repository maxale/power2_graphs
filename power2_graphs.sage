# power2_graphs.sage ver. 20250202 (c) 2022,25 by Max Alekseyev

__doc__ = r'''
Implementation of the algorithms proposed in the paper:

M. A. Alekseyev. "On computing sets of integers with maximum number of pairs summing to powers of 2".
Proceedings of the 35th International Workshop on Combinatorial Algorithms (IWOCA), 2024, Ischia, Italy.
Lecture Notes in Computer Science 14764 (2024): 1–13. doi:10.1007/978-3-031-63021-7_1 arXiv:2303.02872

It provides the following general functions:

* solve_in_2powers(EQ,NZ): solve a given system of equations EQ and inequations NZ in powers of 2.
* graph_solve_2powers(G): find all symbolic vertex labelings (evaluating to distinct integers) of 
                          graph G vertices such that each edge endpoints sum up to a power of 2.
* all_mfs(n): compute all MFSs of order n.
* has_mfs(G): test if a given graph G contains an MFS of order <= 10 (excluding cycle C_4).
* is_admissible(G): test a given graph G for admissibility.
* mag_min_degree(n): return the minimum possible degree of a MAG of order n.
* all_denovo_mags(n): compute denovo MAGs of order `n`.
* mag_extend(n): extend (sub)MAGs of order n-1 to those of order n by adding a new vertex.
and others.

Functions may have optional arguments not mentioned above.

-----------------------

Brief history:
* 20250202 - first public release
'''

# WARNING: solve_in_2powers() is subject to MEMORY LEAK bug in polynomial substitution, see https://github.com/sagemath/sage/issues/27261

import itertools
import functools
import time
from collections import Counter

import multiprocessing
N_CPU = multiprocessing.cpu_count()
print('CPUs:',N_CPU)

# load graph data (MFSs, MAGs, and subMAGs)
attach graphs.dat.py


################################################# Sequence A352178 data
'''
A352178: Let S = {t_1, t_2, ..., t_n} be a set of n distinct integers and consider the sums t_i + t_j (i<j); 
a(n) is the maximum number of such sums that are powers of 2, over all choices for S.

We have a(n) >= A347301(n) for n >= 6, and a(n) <= A006855(n) for all n.
'''

# lower bounds
LB = (0,)*(oeis('A347301').offsets()[0]) + oeis('A347301').first_terms()

# upper bounds
UB = (0,)*(oeis('A006855').offsets()[0]) + oeis('A006855').first_terms()

known_terms = (0,)*(oeis('A352178').offsets()[0]) + oeis('A352178').first_terms()
# known_terms = [0, 0, 1, 3, 4, 6, 7, 9, 11, 13, 15, 17, 19, 21, 24, 26, 29, 31, 34, 36, 39, 41]


################################################# Testing for MFS presence

def has_flower(G):
    '''
    Test if G contains a "flower" MFS.
    '''
    return any( G.subgraph(G.neighbors(v)).matching_polynomial()[d-6] for v in G.vertices() if (d:=G.degree(v)) >= 6 )

def has_saw(G):
    '''
    Test if G contains a "saw" MFS.
    '''
    for u,v in G.edges(labels=False):
        U = set( G.neighbors(u) )
        V = set( G.neighbors(v) )
        C = U & V
        if len(C)==0:
            continue
        L = G.subgraph( U - {v} )
        if L.size()==0:
            continue
        elif L.size()==1:               # not necessary, but speeds up
            C -= set( L.edges(labels=False)[0] )
            if len(C)==0:
                continue
        R = G.subgraph( V - {u} )
        if R.size()==0:
            continue
        elif R.size()==1:               # not necessary, but speeds up
            C -= set( R.edges(labels=False)[0] )
            if len(C)==0:
                continue
        if any( len((s1:=set(e1)) & (s2:=set(e2)))==0 and len(C-s1-s2)!=0 for e1 in L.edges(labels=False) for e2 in R.edges(labels=False) ):
            return True         # saw is formed by (u,v), e1, e2 and any vertex from (C - set(e1) - set(e2))
    return False


assert has_saw(MFS[0]) and has_flower(MFS[1])           # make sure that MFS[0] is a saw and MFS[2] is a flower
MFS10 = MFS[2:]

# check if given graph G contains an MFS as subgraph (in which case it's inadmissible)
def has_mfs(G):
    '''
    Test if G contains an MFS of order <= 10 (excluding cycle C_4).
    '''
    return has_flower(G) or has_saw(G) or any(H.is_subgraph(G, induced=False, up_to_isomorphism=True) for H in MFS10)


def par_has_mfs(g6):
    G = Graph(g6,format='graph6')
    return [] if has_mfs(G) else [g6]

def par_has_mfs7(g6):
    G = Graph(g6,format='graph6')
    return [] if has_flower(G) or has_saw(G) else [g6]


def filter_with_mfs(todo):
    '''
    Given list `todo` of graph6 strings, return a sublist of those graphs that do not contains an MFS
    '''
    res = []
    with multiprocessing.Pool(processes=N_CPU) as pool:
        for k,r in enumerate( pool.imap_unordered(par_has_mfs,todo) ):
            res.extend(r)
            print(f'{k+1} / {len(todo)} : {len(r)} -> {len(res)}',end='   \r',flush=True)
    return res


def filter_with_mfs7(filename):
    '''
    Given a file containing graph6 strings, return a list of those graphs that do not contains an MFS of order 7
    '''
    with open(filename) as f:
        todo = [l.rstrip() for l in f.readlines()]
    res = []
    with multiprocessing.Pool(processes=N_CPU) as pool:
        for k,r in enumerate( pool.imap_unordered(par_has_mfs7,todo) ):
            res.extend(r)
            print(f'{k+1} / {len(todo)} : {len(r)} -> {len(res)}',end='   \r',flush=True)
    return res



######################################################################### Testing for admissibility

def is_admissible(G):
    '''
    Test if a given graph G is admissible.
    '''
    if has_mfs(G):
        return False
    try:
        next(graph_solve_2powers(G))
    except StopIteration:
        return False
    return True


######################################################################## SolveInPowers implementation

def solve_in_2powers(EQ,NZ):
    '''
    Solve a given system of equations EQ and inequations NZ in powers of 2. 
    Both EQ and NZ are sets/lists of polynomials.
    Note that NZ may be changed by this function.
    '''

    #print(f'L: {EQ}\tNZ: {NZ}')

    if not EQ:
        yield dict()
        return

    # we should see both positive and negative coefficients in each element of EQ, in particular the size of each element must be >= 2
    if any( min(r.coefficients())>0 or max(r.coefficients())<0 for r in EQ ):
        return

    D = min( EQ, key=lambda t: t.number_of_terms() )
    assert D.number_of_terms() > 0                      # we assume that there are no zero equations

    if D.number_of_terms()==2 and sum(map(odd_part,D.coefficients())):                   # if there are two elements, they have to cancel each other
        return

    # counts for positive and negative coefficients, pn[-1] + pn[1] = len(D)
    pn = Counter(map(sign,D.coefficients()))

    if pn[-1]==1 and pn[1]>1 and min(c.odd_part() for c in D.coefficients() if c>0) + next(c.odd_part() for c in D.coefficients() if c<0) >= 0:
        D = sum(c*t for c,t in D if c > 0)
    if pn[-1]>1 and pn[1]==1 and max(c.odd_part() for c in D.coefficients() if c<0) + next(c.odd_part() for c in D.coefficients() if c>0) <= 0:
        D = sum(c*t for c,t in D if c < 0)

    #print(f'D: {D}')

    for m in Combinations(D.monomials(),2):

        # compute substitution y -> s + f, where y,s are variables and f is a constant
        y,s = sorted(m, key=lambda t: (valuation(D[t],2),t))
        f = valuation(D[s],2) - valuation(D[y],2)               # notice that f >= 0

        # we set up nu_2(c_y 2^y) = nu_2(c_s 2^s), that is y = s + nu_2(c_s) - nu_2(c_y) = s + f

        #print(f'Subs: {y} -> {s * 2^f}\tX: {X}')

        NZ2 = { r for t in NZ if (r := t.subs({y:s*2^f})).number_of_terms()!=1 }
        if 0 in NZ2:
            continue                    # setting y -> s+f to 0 was processed already
        NZ.add(y - s*2^f)

        EQ2 = [r for e in EQ if not (r := e.subs({y:s*2^f})).is_zero()]

        #print(f'EQ2: {EQ2}\tNZ2:{NZ2}')
        for sol in solve_in_2powers(EQ2,NZ2):
            #print(f'L: {L}\tsol: {sol}\tSubs: {y} -> {s + f}\tEQ2: {EQ2}')
            sol.update({y:s.subs(sol)+f})
            yield sol

    return


######################################################################## GraphSolve implementation

def graph_solve_2powers(G):
    '''
    For a given graph `G`, find all symbolic vertex labelings (evaluating to distinct integers)
    such that each edge endpoints sum up to a power of 2.
    '''

    # G.relabel()               # relabel to make sure vertices are {0, 1, ..., n-1}

    M = Matrix(ZZ, G.size(), G.order())
    for k,e in enumerate(G.edges(labels=False)):
        M[k,e[0]] = 1
        M[k,e[1]] = 1
    #print(f'M: rank={M.rank()} #cols={M.ncols()}\n{M}\n')



    # obtaining non-equalities
    pr = M.pivot_rows()
    pc = M.T.pivot_rows()

    M2 = M[pr,pc].adjugate()
    #print(f'M2:\n{M2}\n')

    #print(f'M2*Mpv:\n{M2*M[pr,pc]}\n')               # this is const * identity_matrix


    # polynomial representation - SHOULD WE USE IT FOR NON-EQUALITIES?
    R = PolynomialRing(ZZ,M.nrows(),'x')
    X = R.gens()

    # constucting a particular solution vector as a polynomial where X[i] stands for 2^x[i]
    N = [0]*M.ncols()
    for k,r in zip(pc,M2.rows()):
        N[k] = sum(e*X[i] for e,i in zip(r,pr))
    #print('N:',N)

    Kr = M.right_kernel().matrix()
    assert Kr.ncols() == len(N)

    NL = set()
    for p,q in Combinations(len(N),2):
        if Kr.column(p)==Kr.column(q):
            d = N[p] - N[q]

            if d.is_zero():
                return                          # no way to make N[p] and N[q] distinct

            d //= gcd(d.coefficients()) * sign( d.lc() )        # normalizing polynomial d

            if d.number_of_terms()<=2 and sum(map(odd_part,d.coefficients())):               # such non-equality always holds
                continue
            NL.add(d)
    #print(f'NL: {len(NL)} {list(map(lambda t: sum(1 for e in t if e),NL))}\t{NL}')


    # constructing equalities

    K = M.left_kernel().matrix().LLL()                  # LLL helps to minimize the number of non-zero coefficients and make them small
    #print(f'K:\n{K}\nabs row sums:{[sum(map(abs,r)) for r in K.rows()]}')

    EQ = [ sum(e*x for e,x in zip(r,X)) for r in K.rows() ]

    #print(f'K:\n{K}\nabs row sums:{[sum(map(abs,r)) for r in K.rows()]}')
    #print(f'CB:\n{G.cycle_basis()}\n')
    #print(f'LLL(K):\n{K.LLL()}\nabs row sums:{[sum(map(abs,r)) for r in K.LLL().rows()]}')

    #assert K*vector(ZZ,[1]*G.size()) == 0

    #print(f'Kernels\t|left|: {K.nrows()}\tright: {Kr}')

    #print(f'Right kernel: {Kr}' )

    M.change_ring(SR)

    #print(EQ[0].parent())
    #print( next(iter(NL)).parent() )
    #print(EQ,NL)

    for sol in solve_in_2powers(EQ,NL):
        r = M.solve_right( vector(SR,(2^SR(x.subs(sol)) for x in X)) )
        #print(f'sol: {sol}\t-->\tr: {r}')
        S = Matrix(SR, [tuple(e.simplify_full() for e in r)] + Kr.rows())
        #print(f'S:\n{S}')

        assert len(set(S.columns())) == len(r)          # elements must be distinct
        yield S
    return


############################################################# Minimal forbidden subgraphs (MFS) generation

'''
def par_mfs0(n,fsg,thread):
    print(f'Thread {thread}: Start')
    res = []
    for G in graphs.nauty_geng(options=f'-c -f -d2 {n} {thread}/{2*N_CPU}'):           # notice that such graphs are connected, which may speed up generation
        if all(not H.is_subgraph(G, induced=False, up_to_isomorphism=True) for H in fsg):
            try:
                next(gsolve_2powers(G))
            except StopIteration:
                res.append(G)
                print(f'Thread {thread}: new MFS {G.graph6_string()}')
    print(f'Thread {thread}: Done!')
    return res

def all_mfs0(U):
    fsg = MFS.copy()
    for n in (10..U):
        with multiprocessing.Pool(processes=N_CPU) as pool:
            new_fsg = sorted( sum(pool.imap_unordered( functools.partial(par_mfs0,n,fsg), range(2*N_CPU) ), []), key=lambda g: g.size() )        # sorted by the number of edges

        new_fsg = [G for k,G in enumerate(new_fsg) if all(H.size()==G.size() or not H.is_subgraph(G, induced=False, up_to_isomorphism=True) for H in new_fsg[:k])]

        print(f"Order {n}: r'{new_fsg}'")
        fsg.extend( new_fsg )
    return [G.graph6_string() for G in fsg]
'''



def par_mfs(fsg,gs6):
    print(f'Task: {len(gs6)} graphs')

    stat = vector(RR,[0,0,0])   # number filetered, processed, time

    res = []
    for G in gs6:
        if all(not H.is_subgraph(G, induced=False, up_to_isomorphism=True) for H in fsg):
            st = time.time()
            try:
                next(graph_solve_2powers(G))
                print('.',end='',flush=True)
            except StopIteration:
                res.append(G)
                print(f'New MFS {G.graph6_string()}')
            stat[2] += time.time() - st
            stat[1] += 1
        else:
            stat[0] += 1
    print('Done:', len(res), stat)
    return res, stat


def all_mfs(n):
    '''
    Compute all MFSs of order n.
    '''
    new_fsg = []
    stat = vector(RR,[0,0,0])

    '''
    todo = list( graphs.nauty_geng(options=f'-c -f -d2 {n}') )
    with multiprocessing.Pool(processes=N_CPU) as pool:
        #new_fsg = sorted( sum(pool.imap_unordered( functools.partial(par_mfs,MFS), (todo[i::N_CPU] for i in range(N_CPU)) ), []), key=lambda g: g.size() )        # sorted by the number of edges
        for gg,ss in pool.imap_unordered( functools.partial(par_mfs,MFS), (todo[i::N_CPU] for i in range(N_CPU)) ): 
            new_fsg.extend(gg)
            stat += ss
    new_fsg = sorted( new_fsg, key=lambda g: g.size() )        # sorted by the number of edges

    print('Candidates:',len(new_fsg),'\tStat::',stat,'\tTime:',stat[2]/stat[1])
    # filter non-minimal graphs
    new_fsg = [G for k,G in enumerate(new_fsg) if all(H.size()==G.size() or not H.is_subgraph(G, induced=False, up_to_isomorphism=True) for H in new_fsg[:k])]
    '''

    # generate in order of increasing size
    for sz in range(n,UB[n]+1):
        todo = list( graphs.nauty_geng(options=f'-c -f -d2 {n} {sz}:{sz}') )
        with multiprocessing.Pool(processes=N_CPU) as pool:
            #new_fsg = sorted( sum(pool.imap_unordered( functools.partial(par_mfs,MFS), (todo[i::N_CPU] for i in range(N_CPU)) ), []), key=lambda g: g.size() )        # sorted by the number of edges
            for gg,ss in pool.imap_unordered( functools.partial(par_mfs,MFS+new_fsg), (todo[i::N_CPU] for i in range(N_CPU)) ): 
                new_fsg.extend(gg)
                stat += ss

    print('Candidates:',len(new_fsg),'\tStat::',stat,'\tTime:',stat[2]/stat[1])

    print(f'Order {n} MFS sizes:', [G.size() for G in new_fsg])
    return [G.graph6_string() for G in new_fsg]



############################################################# Maximum Admissible Graphs (MAGs) generation

def mag_min_degree(n):
    '''
    Return the minimum possible degree of a MAG of order n.
    '''
    # Notice that degree of any vertex cannot be < b - known_terms[n-1]; otherwise removing it will yield a solution for (n-1) with > known_terms[n-1] edges
    mind = known_terms[n] - known_terms[n-1]
    assert n*mind/2 <= known_terms[n]
    return mind


def par_denovo_mags(n,thread,size=None,mind=None,to_test=True):
    res_al = []
    #res_fg = []

    b = known_terms[n] if size is None else size

    if mind is None:
        # Notice that degree of any vertex cannot be < b - known_terms[n-1]; otherwise removing it will yield a solution for (n-1) with > known_terms[n-1] edges
        mind = b - known_terms[n-1]
        assert n*mind/2 <= b

        # check if we can use known MAG; focus on graphs constructed denovo (not from MAG)
        if n-1 in MAG:
            mind += 1
            print(f'Thread {thread}: Incrementing min.degree to {mind}')

    print(f'Thread {thread}: Generating candidate graphs with {b} edges and min.degree {mind}')

    stat = vector(RR,[0,0,0])   # number filetered, processed, time

    for k,G in enumerate( graphs.nauty_geng(options=f'-c -f -d{mind} {n} {b}:{b} {thread}/{N_CPU}') ):           # notice that such graphs are connected, which may speed up generation

        if has_mfs(G):
            print('.',end='',flush=True)
            stat[0] += 1
            continue

        if not to_test:
            print('+',end='',flush=True)
            res_al.append( G.graph6_string() )
            continue

        # processing graphs no MFS
        print(f'Thread {thread}:',G.graph6_string(),'\tSize:', G.size(), G.order())
        stat[1] += 1
        st = time.time()
        try:
            S = next(graph_solve_2powers(G))
        except StopIteration:
            stat[2] += time.time() - st
            print(f'Thread {thread}: No solution for graph {G.graph6_string()}\ttime: {time.time()-st:.2f}s\t[{k}]')
            #res_fg.append( G.graph6_string() )
        else:
            stat[2] += time.time() - st
            print(f'Thread {thread}: a solution for graph {G.graph6_string()} : {S}\ttime: {time.time()-st:.2f}s\t[{k}]')
            res_al.append( G.graph6_string() )
    print(f'Thread {thread}: Done ({len(res_al)}\t{stat}')      # : {len(res_fg)}).')
    return res_al, stat


def all_denovo_mags(n,size=None,mind=None,to_test=True):
    '''
    Compute denovo MAGs of order `n`, size `size` (if given), and minimum degree `mind` (if given).
    When `to_test=False`, admissibility is not tested and computed graphs are only candidate.
    '''
    stat = vector(RR,[0,0,0])   # number filetered, processed, time

    with multiprocessing.Pool(processes=N_CPU) as pool:
        L = []
        for r,ss in pool.imap_unordered( functools.partial(par_denovo_mags,n,size=size,mind=mind,to_test=to_test), range(N_CPU) ):
            L.extend(r)
            print(f'|L|: {len(L)}')
            stat += ss
    print('MAGs:',len(L),'\tStat::',stat,'\tTime:',stat[2]/stat[1])
    return L



def gen_mag_extend(mind, PG, G, to_test=True):
    '''
    Extend known extrenal graphs from n-1 to n.
    `PG` is a shared dict (with dummy values) of processed graphs.
    '''
    # we are attaching a vertex -1 of mind
    # also, degree of any vertex in the resulting graph must be >= mind

    core_neighbors = {v for v in G.vertices() if G.degree(v) < mind}    # vertices that must be connected to a new one
    if len(core_neighbors) > mind or any(G.degree(v) < mind-1 for v in core_neighbors):      # too many core neighbors
        return
    if any( not set(G.neighbors(u)).isdisjoint( set(G.neighbors(v)) ) for u,v in Combinations(core_neighbors,2) ):
        return          # C4 will be present

    for other_neighbors in Subsets(set(G.vertices())-core_neighbors, mind-len(core_neighbors)):
        c = core_neighbors | other_neighbors.set()
        if any( not set(G.neighbors(u)).isdisjoint( set(G.neighbors(v)) ) for u,v in Combinations(c,2) ):
            continue                    # C4 will be present
        H = G.copy()
        H.add_edges((-1,v) for v in c)
        H = H.canonical_label()
        h6 = H.graph6_string()
        if h6 in PG:
            continue

        #PG[h6] = 0      # adding h6 to PG
        PG.update({h6:0})

        # testing
        if has_mfs(H):
            continue

        if not to_test:
            yield h6
            continue

        try:
            S = next(graph_solve_2powers(H))
        except StopIteration:
            print(f'No solution for graph {h6}')
        else:
            print(f'Solution for graph {h6} : {S}')
            yield h6


def par_mag_extend(mind,PG,G,to_test=True):
    return list(gen_mag_extend(mind,PG,G,to_test=to_test))


# note that testing may be subject to memory leak
# alternative route: get a list with to_test=False and process it with process_todo()
def mag_extend(n,mind=None,to_test=True,AG=MAG):
    '''
    Extend (sub)MAGs of order n-1 to those of order n by adding a new vertex (of degree `mind` if given).
    `AG` specify a dict with source graphs (`MAG` or `s_MAG`).
    When `to_test=False`, admissibility is not tested and computed graphs are only candidate.
    '''

    assert n-1 in AG
    mind = mag_min_degree(n) if mind is None else mind
    print(f'Extending {len(AG[n-1])} graphs on {n-1} to {n} nodes by adding a node of degree {mind}')
    with multiprocessing.Pool(processes=N_CPU) as pool:
        with multiprocessing.Manager() as manager:
            #return set( sum(pool.imap_unordered( functools.partial(par_mag_extend, mind, manager.dict()), AG[n-1] ), []) )
            PG = manager.dict()
            res = []
            for k,r in enumerate( pool.imap_unordered( functools.partial(par_mag_extend, mind, PG, to_test=to_test), AG[n-1] ) ):
                print(f'Graph # {k+1} yields {len(r)} graphs',end='   \r',flush=True)
                res.extend(r)
            print('Candidate graphs:', len(PG))
            return res


def extend_with_3to3(todo):
    '''
    Extend graphs from `todo` list to graphs of minimum degree=3 by adding a =3-3= subgraph.
    '''
    print(f'Extending {len(todo)} graphs with =3-3= gadget')
    res = set()
    with multiprocessing.Pool(processes=N_CPU) as pool:
        with multiprocessing.Manager() as manager:
            PG = manager.dict()
            for k,r in enumerate( pool.imap_unordered( functools.partial(extend_with_3to3_1graph,PG), todo )):
                res.update(r)
                print(f'Base {k+1} / {len(todo)}: generated {len(r)} -> {len(res)} / {len(PG)}')
    return res


# processing a single graph
def extend_with_3to3_1graph(PG, G):
    res = set()
    core_neighbors = {v for v in G.vertices() if G.degree(v) < 3}    # vertices that must be connected to a new one
    if sum( 3-G.degree(v) for v in core_neighbors ) > 4:            # deficiency in degrees of core neighbors
        return res

    P = set( Subsets(G.vertices(),2) )     # candidate neignborhoods of new vertices -1 or -2
    for v in G.vertices():
        P.difference_update( Subsets(G.neighbors(v),2) )    # otherwise v, -1/-2, and the two neighbors of v form a 4-cycle
    for p,q in Combinations(P,2):
        if any(G.has_edge(u,v) for u in p for v in q):      # u,v,A,B form a 4-cycle
            continue
        if not all(G.degree(v) + int(v in p) + int(v in q) >= 3 for v in core_neighbors):    # p,q must cover core_neighbors
            continue
        H = G.copy()
        H.add_edges( (u,-1) for u in p )
        H.add_edges( (v,-2) for v in q )
        H.add_edge(-1,-2)
        H = H.canonical_label()
        h6 = H.graph6_string()

        if h6 in PG:
            continue
        PG.update({h6:0})

        # testing
        if not has_mfs(H):
            res.add(h6)
    return res


def denovo_many_deg3(n_nodes, n_edges=None, n_deg3=None):
    '''
    Generate denovo graphs of order `n_nodes` and size `n_edges` (if given) with disconnected `n_deg3` degree-3 vertices.
    If `n_deg3` is not given, the smallest value is autocomputed.
    '''

    if n_edges is None:
        n_edges = known_terms[n_nodes] - 1
        print('Focus on sub-MAGs')
    print(f'Generatings graphs of order {n_nodes} and size {n_edges}')

    n_deg3 = 4*n_nodes - 2*n_edges if n_deg3 is None else n_deg3                # 3*x + 4*(n_nodes-x) >= 2*n_edges
    print(f'...by adding {n_deg3} nodes of degree 3')

    if 3*n_deg3 > n_edges:
        print('No such graphs')
        return

    # base graphs are don't have to be connected 
    GG = list( graphs.nauty_geng(options=f'-f {n_nodes-n_deg3} {n_edges-3*n_deg3}:{n_edges-3*n_deg3}') )
    GG = filter_with_mfs(GG)

    res = set()
    with multiprocessing.Pool(processes=N_CPU) as pool:
        for k,G in enumerate( pool.imap_unordered( functools.partial(denovo_many_deg3_1graph,n_deg3=n_deg3), GG )):
            res.update(G)
            print(f'Base {k+1} / {len(GG)}: generated {len(G)} -> {len(res)}')
    return res


def denovo_many_deg3_1graph(G, n_deg3):
    V = set( G.vertices() )
    P = set( Subsets(V,3) )     # candidate neignborhoods of new vertices of degree 3
    for v in V:
        for u2 in Subsets(G.neighbors(v),2):
            P.difference_update( u2|Set({u3}) for u3 in V-{*u2} )
    return set( gen_denovo_many_deg3(G,list(P),n_deg3) )


def gen_denovo_many_deg3(G, P, to_add, st=0):
    if to_add==0:
        if any(G.degree(u)==3 and G.degree(v)==3 for u,v in G.edges(labels=False)):     # contains =3-3=
            return
        if G.is_connected() and min(G.degree())==3:
            yield G.canonical_label().graph6_string()
        return

    s = sum( max(0,3-d) for d in G.degree() )           # deficiency in degrees
    _3to3_ = sum((e for e in G.edges(labels=False) if G.degree(e[0])==3 and G.degree(e[1])==3), tuple())     # number of =3-3= gadgets = len(_3to3_)//2
    cnt = sorted((_3to3_.count(v) for v in set(_3to3_)), reverse=True)
    t = 0
    while sum(cnt[:t]) < len(_3to3_)//2:
        t += 1
    # t <= minimum number of vertices whose degree must be changed to destroy all =3-3= gadgets

    if  s+t > 3*to_add:
        return

    for i in range(st,len(P)):
        if any(len(set(G.neighbors(u)) & set(G.neighbors(v))) for u,v in Subsets(P[i],2)):
            continue
        H = G.copy()
        H.add_edges( (-i-1,v) for v in P[i] )
        yield from gen_denovo_many_deg3(H, P, to_add-1, i+1)



################################# Workaround for the memory leak in all_denovo_mags() by processing each graph separately

def par_todo_1graph(s6):
        G = Graph( s6.rstrip(), format='graph6' )
        st = time.time()
        try:
            S = next(graph_solve_2powers(G))
        except StopIteration:
            return None, time.time() - st
        return s6, time.time() - st


def par_proc_todo(L):
    print(f'Task: {len(L)} graphs')
    res = []
    for g6 in L:
        try:
            S = next(graph_solve_2powers(Graph(g6,format='graph6')))
        except StopIteration:
            print('.',end='')
        else:
            res.append( g6 )
            print(f'Solution for graph {g6} : {S}')
    print('Done:', len(res))
    return res

# attach filtered_eg15.py

# test graphs in chunks using multiprocessing, each chunk should not leak too much memory (which is recovered when chunk processing is done)
# todo contains list of MFS-free graphs
# suggested: per_thread=8 is too much for order 17,18; per_thread=7 for order 19,20
def process_todo(todo,per_thread=9):
    chunk_size = per_thread * N_CPU

    '''
    res = []
    for s in range(0,len(todo15),chunk_size):
        with multiprocessing.Pool(processes=N_CPU) as pool:
            res.extend( sum(pool.imap_unordered( par_proc_todo, (todo15[s+i:s+chunk_size:N_CPU] for i in range(N_CPU)) ), []) )
        print(f'CHUNK {s}: {len(res)}')
    return res
    '''

    TD = len(todo)
    print(f'Testing {TD} graphs')

    stat = vector(RR,[0,0,0])   # number filetered, processed, time

    L = []
    tot_time = 0
    tot_k = 0

    for s in range(0,TD,chunk_size):
      print(f'\nChunk # {s//chunk_size + 1} / {TD//chunk_size + 1}')
      with multiprocessing.Pool(processes=N_CPU) as pool:
        for r,ss in pool.imap_unordered( par_todo_1graph, todo[s:s + chunk_size] ):
            tot_time += ss
            tot_k += 1
            print(f'Done {tot_k*100./TD:.2f}%   Avg. time: {tot_time/tot_k:.1f}s       ',end='\r',flush=True)
            if r:
                L.append(r)
                print(f'\n==> {r}',len(L))
    print('\nMAGs:',len(L),'\tTime:',tot_time/TD)
    return L



# all_denovo_mags() causes memory leak
def process_file_nomfs(filename='candidates_20_40_no_mfs10.txt',per_thread=9):
    with open(filename) as f:
    #with open('mag_denovo_17_31_no_mfs10.txt') as f:
        todo = [l.rstrip() for l in f.readlines()]

    #chunk_size = 7*N_CPU               # = 280 graphs (used for n=20)
    #chunk_size = 9*N_CPU               # = 360 graphs (used for n=17)

    return process_todo(todo,per_thread=per_thread)



################################################################## Auxiliary functions

def graph_from_numbers(L):
    G = Graph()
    for v1,v2 in Combinations(L,2):
        s = v1 + v2
        if s >= 1 and s == 1<<valuation(s,2):
            G.add_edge(v1,v2)
    return G

#list( gsolve_2powers( graph_from_numbers( [  -9, -7, -5, -3, -1, 1, 3,  5,  7,  9, 11, 13 ] ) ) )


'''
From this we can conclude that all cycles in the graph G must contain at least one negative number..
'''
def test_conjecture(L):
    G = graph_from_numbers(L)
    C = [frozenset(tuple(sorted(e[:2])) for e in c) for c in G.cycle_basis(output='edge')]
    for S in Subsets(C):
        T = set()
        for c in S:
            T = T.symmetric_difference(c)
        if min((min(e) for e in T),default=-1)>0:
            return False, T
    return True


def debug():
    K = Matrix( [(1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, -1, 0, 0, 1), (0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, -1, 0, 0, 1), (-1, 0, 1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0), (0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 1, 0, 0, -1, 0, 1), (-1, 1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, -1, 1, 1, -1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, -1, 0, 1, 0, 0, 0, -1, 1), (0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 1, 0, -1, 1, 0, 0, 0)] )
    for sol in matsolve_2powers(K):
        if K*sol == 0:
            print(f'Sol: {sol}')
        else:
            print('Bad:', sol)
            return


def plot_labeled_graph(G):
    L = next( graph_solve_2powers(G) )
    assert L.nrows() == 1
    L = L.row(0)
    L /= gcd(L)
    #G.set_pos(G.layout_circular())
    #return G.graphplot(vertex_labels=lambda i: L[i], vertex_size=500, figsize=[8,8])
    #G.show(vertex_labels=lambda i: L[i], vertex_size=400, figsize=[8,8], spring=True, layout='circular')
    # return G.graphplot(vertex_labels=lambda i: L[i], edge_labels=lambda x,y: f"$2^{valuation(x+y,2)}$", vertex_size=500, fontsize=16, figsize=[4,4], layout='graphviz', prog='circo').plot()
    return G.graphplot(vertex_labels=lambda i: L[i], vertex_size=400, fontsize=20, figsize=[6,6], layout='graphviz', prog='circo').plot()


'''
# used to draw graphs in SageCell
for G in MFS:
    #G.is_planar(set_embedding=True)
    G.show(vertex_labels=False, vertex_size=200, fontsize=16, figsize=[4,4], layout='spring', spring=True)
'''




