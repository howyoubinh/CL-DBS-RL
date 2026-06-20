import numpy as np
from scipy.signal.windows import dpss
from scipy.interpolate import interp1d
from scipy.stats import chi2, t
from scipy.io import savemat
from scipy.stats import norm
from scipy.special import expit

from timeit import default_timer as timer

from numba import jit, njit
from numba.typed import Dict

import os

def simulate_network_opt(IT, pd, corstim, n=10, tmax=2000, dt=0.01, PW=0.3, amplitude=300, dbs_freq=100, states={}, seed=None, path=None):
    """
    This program simulates the rat model of Parkinson's disease originially implemented in Matlab

    Input parameters:
    IT - iteration number (trial no)
    pd - 0(normal/healthy condition), 1(Parkinson's disease(PD) condition)
    corstim (cortical stimulation) - 0(off), 1(on) 
    n - number of neurons in each nucleus
    tmax - length of signal to simulate (ms) [1000 = 1 second]
    dt - time step (default: 0.01 ms)
    PW - DBS pulse width
    amplitude - DBS step amplitude
    dbs_freq - DBS frequency (0 is off)  
    states - dictionary to save state variables
    seed - seed value
    path - location of .mat file saved
    """

    if seed is not None:
        np.random.seed(int(seed))
    else:
        np.random.seed()

    t = np.arange(0, tmax + dt, dt)

    # DBS Parameters

    pattern = dbs_freq # Hz

    # Create DBS Current (currently it is on STN (check Idbs))

    if dbs_freq == 0:
        Idbs = np.zeros(len(t))
    else:
        Idbs = creatdbs(pattern, tmax, dt, PW, amplitude)

    # Create Cortical Stimulus Pulse
    if corstim == 1:
        Iappco = np.zeros(len(t))
        Iappco[int(1000/dt):int((1000+0.3)/dt)] = 350
    else:
        Iappco = np.zeros(len(t))

    # Run CTX-BG-TH Network Model
    TH_APs, STN_APs, GPe_APs, GPi_APs, Striat_APs_indr, Striat_APs_dr, Cor_APs, states = CTX_BG_TH_network_CL(pd, corstim, tmax, dt, n, Idbs, Iappco, states)

    # Calculate GPi pathological low-frequency oscillatory power
    dt1 = 0.01 * 10**-3 # convert to seconds
    params = {
        'Fs': 1/dt1,        # Hz
        'fpass': [1, 100],
        'tapers': [3, 5],
        'trialave': 1
    }
    gpi_alpha_beta_area, gpi_S, gpi_f = make_Spectrum(GPi_APs, params)

    # gpi_alpha_beta_area - GPi spectral power integrated in 7-35Hz band
    # gpi_S - GPi spectral power
    # gpi_f - GPi spectral frequencies

    # Save the results
    if pd == 0:
        name = f'{IT}con_f{int(pattern)}_a{int(amplitude)}_pw{PW}py.mat'
    else:
        name = f'{IT}pd_f{int(pattern)}_a{int(amplitude)}_pw{PW}py.mat'
    vars_to_save = {
        'amplitude': amplitude, 
        'Cor_APs': Cor_APs,
        'corstim': corstim,
        'dt': dt,
        'dt1': dt1,
        'GPe_APs': GPe_APs,
        'gpi_alpha_beta_area': gpi_alpha_beta_area,
        'GPi_APs': GPi_APs,
        'gpi_f': gpi_f.tolist(),
        'gpi_S': gpi_S.tolist(),
        'Iappco': Iappco.tolist(),
        'Idbs': Idbs.tolist(),
        'IT': IT,
        'n': n,
        'name': name,
        'params': params,
        'pattern': pattern,
        'pd': pd,
        'dbs_freq': dbs_freq,
        'PW': PW,
        'STN_APs': STN_APs,
        'Striat_APs_dr': Striat_APs_dr,
        'Striat_APs_indr': Striat_APs_indr,
        'states': states,
        't': t.tolist(),
        'TH_APs': TH_APs,
        'tmax': tmax
    }
    # folder location

    if path is not None:
        name = os.path.join(path, name)

        savemat(f'{name}', vars_to_save, do_compression=True)
        print(f'Done saving file: {name}')

    


    # or use scipy to save at .mat file
    return vars_to_save

# @jit(nopython=True)

@njit()
def _CTX_BG_TH_network_CL_core(
    pd, corstim, tmax, dt, n, Idbs, Iappco,
    vth_0, vsn_0, vge_0, vgi_0, vstr_indr_0, vstr_dr_0, ve_0, ue_0, vi_0, ui_0,
    N3_0, N4_0, H1_0, H3_0, H4_0, R1_0, R3_0, R4_0, CA2_0, CA3_0, CA4_0,
    N2_0, H2_0, M2_0, A2_0, B2_0, C2_0, D2_0, D1_0, P2_0, Q2_0, R2_0,
    m5_0, h5_0, n5_0, p5_0, m6_0, h6_0, n6_0, p6_0,
    CAsn2_0,
    all_idx, bll_idx, cll_idx, dll_idx, ell_idx, fll_idx, gll_idx, hll_idx, ill_idx, jll_idx, kll_idx, lll_idx, mll_idx, nll_idx, oll_idx,
    gcorsna, gcorsnn, gcordrstr, ggege,
    gsngen, gsngea, gsngi,
    r_p1, r_p2, r_m1, r_m2, r_m3, r_m4, r_m5, r_m6, r_m7, r_m8, r_m9,
    syn_func_th, syn_func_stn_gpea, syn_func_stn_gpen, syn_func_stn_gpi, syn_func_gpe_stn, syn_func_gpe_gpi, syn_func_gpe_gpe, syn_func_gpi_th, syn_func_str_indr, syn_func_str_dr, syn_func_cor_d2, syn_func_cor_stn_a, syn_func_cor_stn_n
):
    t_len = len(Idbs)
    MAX_SPIKES = 200
    t_a = 1000
    t_a_steps = int(t_a / dt)

    vth = np.zeros((n, t_len)); vsn = np.zeros((n, t_len)); vge = np.zeros((n, t_len)); vgi = np.zeros((n, t_len))
    vstr_indr = np.zeros((n, t_len)); vstr_dr = np.zeros((n, t_len)); ve = np.zeros((n, t_len)); vi = np.zeros((n, t_len))
    ue = np.zeros((n, t_len)); ui = np.zeros((n, t_len))

    vth[:, 0] = vth_0; vsn[:, 0] = vsn_0; vge[:, 0] = vge_0; vgi[:, 0] = vgi_0
    vstr_indr[:, 0] = vstr_indr_0; vstr_dr[:, 0] = vstr_dr_0; ve[:, 0] = ve_0; ue[:, 0] = ue_0
    vi[:, 0] = vi_0; ui[:, 0] = ui_0

    N3, N4, H1, H3, H4, R1, R3, R4 = N3_0, N4_0, H1_0, H3_0, H4_0, R1_0, R3_0, R4_0
    CA2, CA3, CA4 = CA2_0, CA3_0, CA4_0
    N2 = N2_0; H2 = H2_0; M2 = M2_0; A2 = A2_0; B2 = B2_0; C2 = C2_0; D2 = D2_0; D1 = D1_0; P2 = P2_0; Q2 = Q2_0; R2 = R2_0
    m5, h5, n5, p5, m6, h6, n6, p6 = m5_0, h5_0, n5_0, p5_0, m6_0, h6_0, n6_0, p6_0
    CAsn2 = CAsn2_0

    ae, be, ce, de = 0.02, 0.2, -65.0, 8.0
    ai, bi, ci, di = 0.1, 0.2, -65.0, 2.0
    Cm = 1.0
    gl = np.array([0.05, 0.35, 0.1, 0.1]); El = np.array([-70.0, -60.0, -65.0, -67.0])
    gna = np.array([3.0, 49.0, 120.0, 100.0]); Ena = np.array([50.0, 60.0, 55.0, 50.0])
    gk = np.array([5.0, 57.0, 30.0, 80.0]); Ek = np.array([-75.0, -90.0, -80.0, -100.0])
    gt = np.array([5.0, 5.0, 0.5]); Et = 0.0
    gca = np.array([0.0, 2.0, 0.15]); Eca = np.array([0.0, 140.0, 120.0])
    Em = -100.0
    gahp = np.array([0.0, 20.0, 10.0]); k1 = np.array([0.0, 15.0, 10.0]); kca = np.array([0.0, 22.5, 15.0])
    ga, gL, gcak, Kca, Z, F, Cao, R, T = 5.0, 15.0, 1.0, 2e-3, 2.0, 96485.0, 2000.0, 8314.0, 298.0
    alp = 1.0 / (Z * F); con = (R * T) / (Z * F)
    Esyn = np.array([-85.0, 0.0, -85.0, 0.0, -85.0, -85.0, -80.0])
    tau, gpeak, gpeak1 = 5.0, 0.43, 0.3
    ggith, ggesn, gstrgpe, gstrgpi, ggigi, gm, ggaba, gcorindrstr, gie, gthcor, gei = 0.112, 0.5, 0.5, 0.5, 0.5, 1.0, 0.1, 0.07, 0.2, 0.15, 0.1
    tau_i = 13.0

    S2a = np.zeros(n); S2b = np.zeros(n); S2an = np.zeros(n)
    S3a = np.zeros(n); S3b = np.zeros(n); S3c = np.zeros(n)
    S4 = np.zeros(n); S5 = np.zeros(n); S9 = np.zeros(n)
    S6a = np.zeros(n); S6b = np.zeros(n); S6bn = np.zeros(n)
    S7 = np.zeros(n); S8 = np.zeros(n); S1a = np.zeros(n); S1b = np.zeros(n); S1c = np.zeros(n)
    Z1a = np.zeros(n); Z1b = np.zeros(n)

    sp_th = np.full((n, MAX_SPIKES), -1, dtype=np.int64); sc_th = np.zeros(n, dtype=np.int64)
    sp_cor = np.full((n, MAX_SPIKES), -1, dtype=np.int64); sc_cor = np.zeros(n, dtype=np.int64)
    sp_str_indr = np.full((n, MAX_SPIKES), -1, dtype=np.int64); sc_str_indr = np.zeros(n, dtype=np.int64)
    sp_str_dr = np.full((n, MAX_SPIKES), -1, dtype=np.int64); sc_str_dr = np.zeros(n, dtype=np.int64)
    sp_stn = np.full((n, MAX_SPIKES), -1, dtype=np.int64); sc_stn = np.zeros(n, dtype=np.int64)
    sp_gpe = np.full((n, MAX_SPIKES), -1, dtype=np.int64); sc_gpe = np.zeros(n, dtype=np.int64)
    sp_gpi = np.full((n, MAX_SPIKES), -1, dtype=np.int64); sc_gpi = np.zeros(n, dtype=np.int64)

    for i in range(1, t_len):
        V1, V2, V3, V4, V5, V6, V7, V8 = vth[:,i-1], vsn[:,i-1], vge[:,i-1], vgi[:,i-1], vstr_indr[:,i-1], vstr_dr[:,i-1], ve[:,i-1], vi[:,i-1]
        S21a, S21an, S21b, S31a, S31b, S31c, S32c, S32b = S2a[r_p1], S2an[r_p1], S2b[r_p1], S3a[r_m1], S3b[r_m1], S3c[r_m1], S3c[r_p2], S3b[r_p2]
        S11cr, S12cr, S13cr, S14cr = S1c[all_idx], S1c[bll_idx], S1c[cll_idx], S1c[dll_idx]
        S11br, S12br, S13br, S14br = S1b[ell_idx], S1b[fll_idx], S1b[gll_idx], S1b[hll_idx]
        S11ar, S12ar, S13ar, S14ar = S1a[ill_idx], S1a[jll_idx], S1a[kll_idx], S1a[lll_idx]
        S81r, S82r, S83r = S8[mll_idx], S8[nll_idx], S8[oll_idx]
        S51, S52, S53, S54, S55, S56, S57, S58, S59 = S5[r_m1], S5[r_m2], S5[r_m3], S5[r_m4], S5[r_m5], S5[r_m6], S5[r_m7], S5[r_m8], S5[r_m9]
        S61b, S61bn, S91, S92, S93, S94, S95, S96, S97, S98, S99 = S6b[r_m1], S6bn[r_m1], S9[r_m1], S9[r_m2], S9[r_m3], S9[r_m4], S9[r_m5], S9[r_m6], S9[r_m7], S9[r_m8], S9[r_m9]

        m1, m3, m4, n3, n4, h1, h3, h4, p1, a3, a4, s3, s4, r1, r3, r4 = th_minf(V1), gpe_minf(V3), gpe_minf(V4), gpe_ninf(V3), gpe_ninf(V4), th_hinf(V1), gpe_hinf(V3), gpe_hinf(V4), th_pinf(V1), gpe_ainf(V3), gpe_ainf(V4), gpe_sinf(V3), gpe_sinf(V4), th_rinf(V1), gpe_rinf(V3), gpe_rinf(V4)
        tn3, tn4, th1, th3, th4, tr1, tr3, tr4 = gpe_taun(V3), gpe_taun(V4), th_tauh(V1), gpe_tauh(V3), gpe_tauh(V4), th_taur(V1), 30.0, 30.0
        n2, m2, h2, a2, b2, c2, d2, d1, p2, q2, r2 = stn_ninf(V2), stn_minf(V2), stn_hinf(V2), stn_ainf(V2), stn_binf(V2), stn_cinf(V2), stn_d2inf(CAsn2), stn_d1inf(V2), stn_pinf(V2), stn_qinf(V2), stn_rinf(CAsn2)
        td2, tr2, tn2, tm2, th2, ta2, tb2, tc2, td1, tp2, tq2 = 130.0, 2.0, stn_taun(V2), stn_taum(V2), stn_tauh(V2), stn_taua(V2), stn_taub(V2), stn_tauc(V2), stn_taud1(V2), stn_taup(V2), stn_tauq(V2)

        Ecasn = con * np.log(Cao / CAsn2)
        Il1, Ina1, Ik1, It1, Igith, Iappth = gl[0]*(V1-El[0]), gna[0]*(m1**3)*H1*(V1-Ena[0]), gk[0]*((0.75*(1-H1))**4)*(V1-Ek[0]), gt[0]*(p1**2)*R1*(V1-Et), ggith*(V1-Esyn[5])*S4, 1.2
        vth[:, i] = V1 + dt * (1/Cm * (-Il1 - Ina1 - Ik1 - It1 - Igith + Iappth))
        H1 += dt * ((h1 - H1) / th1); R1 += dt * ((r1 - R1) / tr1)

        for j in range(n):
            if vth[j, i-1] < -10 and vth[j, i] > -10: sp_th[j, sc_th[j]] = 0; sc_th[j] += 1
            tot = 0.0
            for k in range(sc_th[j]): tot += syn_func_th[sp_th[j, k]]
            S7[j] = tot
            if sc_th[j] > 0:
                for k in range(sc_th[j]): sp_th[j, k] += 1
                if sp_th[j, 0] == t_a_steps:
                    for k in range(sc_th[j]-1): sp_th[j, k] = sp_th[j, k+1]
                    sp_th[j, sc_th[j]-1] = -1; sc_th[j] -= 1

        Ina2, Ik2, Ia2, IL2, It2, Icak2, Il2 = gna[1]*(M2**3)*H2*(V2-Ena[1]), gk[1]*(N2**4)*(V2-Ek[1]), ga*(A2**2)*B2*(V2-Ek[1]), gL*(C2**2)*D1*D2*(V2-Ecasn), gt[1]*(P2**2)*Q2*(V2-Ecasn), gcak*(R2**2)*(V2-Ek[1]), gl[1]*(V2-El[1])
        Igesn = ggesn*((V2-Esyn[0])*(S3a+S31a))
        Icorsnampa = gcorsna.T*(V2-Esyn[1])*(S6b+S61b); Icorsnnmda = gcorsnn.T*(V2-Esyn[1])*(S6bn+S61bn)
        vsn[:, i] = V2 + dt * (1/Cm * (-Ina2 - Ik2 - Ia2 - IL2 - It2 - Icak2 - Il2 - Igesn - Icorsnampa - Icorsnnmda + Idbs[i]))
        N2 += dt*((n2-N2)/tn2); H2 += dt*((h2-H2)/th2); M2 += dt*((m2-M2)/tm2); A2 += dt*((a2-A2)/ta2); B2 += dt*((b2-B2)/tb2); C2 += dt*((c2-C2)/tc2); D2 += dt*((d2-D2)/td2); D1 += dt*((d1-D1)/td1); P2 += dt*((p2-P2)/tp2); Q2 += dt*((q2-Q2)/tq2); R2 += dt*((r2-R2)/tr2); CAsn2 += dt * ((-alp * (IL2 + It2)) - (Kca * CAsn2))

        for j in range(n):
            if vsn[j, i-1] < -10 and vsn[j, i] > -10: sp_stn[j, sc_stn[j]] = 0; sc_stn[j] += 1
            ta, tb, tc = 0.0, 0.0, 0.0
            for k in range(sc_stn[j]): idx = sp_stn[j,k]; ta += syn_func_stn_gpea[idx]; tb += syn_func_stn_gpen[idx]; tc += syn_func_stn_gpi[idx]
            S2a[j], S2an[j], S2b[j] = ta, tb, tc
            if sc_stn[j] > 0:
                for k in range(sc_stn[j]): sp_stn[j, k] += 1
                if sp_stn[j, 0] == t_a_steps:
                    for k in range(sc_stn[j]-1): sp_stn[j, k] = sp_stn[j, k+1]
                    sp_stn[j, sc_stn[j]-1] = -1; sc_stn[j] -= 1

        Il3, Ik3, Ina3, It3, Ica3, Iahp3 = gl[2]*(V3-El[2]), gk[2]*(N3**4)*(V3-Ek[2]), gna[2]*(m3**3)*H3*(V3-Ena[2]), gt[2]*(a3**3)*R3*(V3-Eca[2]), gca[2]*(s3**2)*(V3-Eca[2]), gahp[2]*(V3-Ek[2])*(CA3/(CA3+k1[2]))
        Isngeampa = gsngea.T*((V3-Esyn[1])*(S2a+S21a)); Isngenmda = gsngen.T*((V3-Esyn[1])*(S2an+S21an))
        Igege = (0.25*(pd*3+1))*ggege.T*((V3-Esyn[2])*(S31c+S32c)); Istrgpe = gstrgpe*(V3-Esyn[5])*(S5+S51+S52+S53+S54+S55+S56+S57+S58+S59); Iappgpe = 3 - 2 * corstim * (1.0 - pd)
        vge[:, i] = V3 + dt * (1/Cm * (-Il3 - Ik3 - Ina3 - It3 - Ica3 - Iahp3 - Isngeampa - Isngenmda - Igege - Istrgpe + Iappgpe))
        N3 += dt*(0.1*(n3-N3)/tn3); H3 += dt*(0.05*(h3-H3)/th3); R3 += dt*(1*(r3-R3)/tr3); CA3 += dt*(1e-4*(-Ica3 - It3 - kca[2]*CA3))
        for j in range(n):
            if vge[j, i-1] < -10 and vge[j, i] > -10: sp_gpe[j, sc_gpe[j]] = 0; sc_gpe[j] += 1
            ta, tb, tc = 0.0, 0.0, 0.0
            for k in range(sc_gpe[j]): idx = sp_gpe[j,k]; ta += syn_func_gpe_stn[idx]; tb += syn_func_gpe_gpi[idx]; tc += syn_func_gpe_gpe[idx]
            S3a[j], S3b[j], S3c[j] = ta, tb, tc
            if sc_gpe[j] > 0:
                for k in range(sc_gpe[j]): sp_gpe[j, k] += 1
                if sp_gpe[j, 0] == t_a_steps:
                    for k in range(sc_gpe[j]-1): sp_gpe[j, k] = sp_gpe[j, k+1]
                    sp_gpe[j, sc_gpe[j]-1] = -1; sc_gpe[j] -= 1

        Il4, Ik4, Ina4, It4, Ica4, Iahp4 = gl[2]*(V4-El[2]), gk[2]*(N4**4)*(V4-Ek[2]), gna[2]*(m4**3)*H4*(V4-Ena[2]), gt[2]*(a4**3)*R4*(V4-Eca[2]), gca[2]*(s4**2)*(V4-Eca[2]), gahp[2]*(V4-Ek[2])*(CA4/(CA4+k1[2]))
        Isngi = gsngi*((V4-Esyn[3])*(S2b+S21b)); Igigi = ggigi*((V4-Esyn[4])*(S31b+S32b)); Istrgpi = gstrgpi*(V4-Esyn[5])*(S9+S91+S92+S93+S94+S95+S96+S97+S98+S99)
        vgi[:, i] = V4 + dt * (1/Cm * (-Il4 - Ik4 - Ina4 - It4 - Ica4 - Iahp4 - Isngi - Igigi - Istrgpi + 3.0))
        N4 += dt*(0.1*(n4-N4)/tn4); H4 += dt*(0.05*(h4-H4)/th4); R4 += dt*(1*(r4-R4)/tr4); CA4 += dt*(1e-4*(-Ica4-It4-kca[2]*CA4))
        for j in range(n):
            if vgi[j, i-1] < -10 and vgi[j, i] > -10: sp_gpi[j, sc_gpi[j]] = 0; sc_gpi[j] += 1
            tot = 0.0
            for k in range(sc_gpi[j]): tot += syn_func_gpi_th[sp_gpi[j, k]]
            S4[j] = tot
            if sc_gpi[j] > 0:
                for k in range(sc_gpi[j]): sp_gpi[j, k] += 1
                if sp_gpi[j, 0] == t_a_steps:
                    for k in range(sc_gpi[j]-1): sp_gpi[j, k] = sp_gpi[j, k+1]
                    sp_gpi[j, sc_gpi[j]-1] = -1; sc_gpi[j] -= 1

        Ina5, Ik5, Il5, Im5 = gna[3]*(m5**3)*h5*(V5-Ena[3]), gk[3]*(n5**4)*(V5-Ek[3]), gl[3]*(V5-El[3]), (2.6-1.1*pd)*gm*p5*(V5-Em)
        Igaba5 = (ggaba/4)*(V5-Esyn[6])*(S11cr+S12cr+S13cr+S14cr); Icorstr5 = gcorindrstr*(V5-Esyn[1])*S6a
        vstr_indr[:, i] = V5 + (dt/Cm)*(-Ina5-Ik5-Il5-Im5-Igaba5-Icorstr5)
        m5 += dt*(alpham(V5)*(1-m5)-betam(V5)*m5); h5 += dt*(alphah(V5)*(1-h5)-betah(V5)*h5); n5 += dt*(alphan(V5)*(1-n5)-betan(V5)*n5); p5 += dt*(alphap(V5)*(1-p5)-betap(V5)*p5); S1c += dt*((Ggaba(V5)*(1-S1c))-(S1c/tau_i))
        for j in range(n):
            if vstr_indr[j, i-1] < -10 and vstr_indr[j, i] > -10: sp_str_indr[j, sc_str_indr[j]] = 0; sc_str_indr[j] += 1
            tot = 0.0
            for k in range(sc_str_indr[j]): tot += syn_func_str_indr[sp_str_indr[j, k]]
            S5[j] = tot
            if sc_str_indr[j] > 0:
                for k in range(sc_str_indr[j]): sp_str_indr[j, k] += 1
                if sp_str_indr[j, 0] == t_a_steps:
                    for k in range(sc_str_indr[j]-1): sp_str_indr[j, k] = sp_str_indr[j, k+1]
                    sp_str_indr[j, sc_str_indr[j]-1] = -1; sc_str_indr[j] -= 1

        Ina6, Ik6, Il6, Im6 = gna[3]*(m6**3)*h6*(V6-Ena[3]), gk[3]*(n6**4)*(V6-Ek[3]), gl[3]*(V6-El[3]), (2.6-1.1*pd)*gm*p6*(V6-Em)
        Igaba6 = (ggaba/3)*(V6-Esyn[6])*(S81r+S82r+S83r); Icorstr6 = gcordrstr.T*(V6-Esyn[1])*S6a
        vstr_dr[:, i] = V6 + (dt/Cm)*(-Ina6-Ik6-Il6-Im6-Igaba6-Icorstr6)
        m6 += dt*(alpham(V6)*(1-m6)-betam(V6)*m6); h6 += dt*(alphah(V6)*(1-h6)-betah(V6)*h6); n6 += dt*(alphan(V6)*(1-n6)-betan(V6)*n6); p6 += dt*(alphap(V6)*(1-p6)-betap(V6)*p6); S8 += dt*((Ggaba(V6)*(1-S8))-(S8/tau_i))
        for j in range(n):
            if vstr_dr[j, i-1] < -10 and vstr_dr[j, i] > -10: sp_str_dr[j, sc_str_dr[j]] = 0; sc_str_dr[j] += 1
            tot = 0.0
            for k in range(sc_str_dr[j]): tot += syn_func_str_dr[sp_str_dr[j, k]]
            S9[j] = tot
            if sc_str_dr[j] > 0:
                for k in range(sc_str_dr[j]): sp_str_dr[j, k] += 1
                if sp_str_dr[j, 0] == t_a_steps:
                    for k in range(sc_str_dr[j]-1): sp_str_dr[j, k] = sp_str_dr[j, k+1]
                    sp_str_dr[j, sc_str_dr[j]-1] = -1; sc_str_dr[j] -= 1

        Iie, Ithcor = gie*(V7-Esyn[0])*(S11br+S12br+S13br+S14br), gthcor*(V7-Esyn[1])*S7
        ve[:, i] = V7 + dt*((0.04*(V7**2)) + (5*V7) + 140 - ue[:, i-1] - Iie - Ithcor + Iappco[i])
        ue[:, i] = ue[:, i-1] + dt*(ae*((be*V7)-ue[:,i-1]))
        for j in range(n):
            if ve[j, i-1] >= 30: ve[j, i], ue[j, i] = ce, ue[j, i-1] + de; sp_cor[j, sc_cor[j]] = 0; sc_cor[j] += 1
            ta, tb, tc = 0.0, 0.0, 0.0
            for k in range(sc_cor[j]): idx = sp_cor[j,k]; ta += syn_func_cor_d2[idx]; tb += syn_func_cor_stn_a[idx]; tc += syn_func_cor_stn_n[idx]
            S6a[j], S6b[j], S6bn[j] = ta, tb, tc
            if sc_cor[j] > 0:
                for k in range(sc_cor[j]): sp_cor[j, k] += 1
                if sp_cor[j, 0] == t_a_steps:
                    for k in range(sc_cor[j]-1): sp_cor[j, k] = sp_cor[j, k+1]
                    sp_cor[j, sc_cor[j]-1] = -1; sc_cor[j] -= 1

        uce = np.zeros(n)
        for j in range(n):
            if ve[j,i-1] < -10 and ve[j,i] > -10: uce[j] = gpeak / (tau * np.exp(-1)) / dt
        S1a += dt * Z1a; Z1a += dt * (uce - 2/tau * Z1a - 1/(tau**2) * S1a)

        Iei = gei*(V8-Esyn[1])*(S11ar+S12ar+S13ar+S14ar)
        vi[:, i] = V8 + dt*(0.04*V8**2+5*V8+140-ui[:, i-1] - Iei + Iappco[i])
        ui[:, i] = ui[:, i-1] + dt*(ai*(bi*V8-ui[:, i-1]))
        for j in range(n):
            if vi[j, i-1] >= 30: vi[j, i], ui[j, i] = ci, ui[j, i-1] + di

        uci = np.zeros(n)
        for j in range(n):
            if vi[j, i-1] < -10 and vi[j, i] > -10: uci[j] = gpeak / (tau * np.exp(-1)) / dt
        S1b += dt * Z1b; Z1b += dt * (uci - 2/tau * Z1b - 1/(tau**2) * S1b)

    return vth, vsn, vge, vgi, vstr_indr, vstr_dr, ve, vi, ue, ui, \
           N3, N4, H1, H3, H4, R1, R3, R4, CA2, CA3, CA4, \
           N2, H2, M2, A2, B2, C2, D2, D1, P2, Q2, R2, \
           m5, h5, n5, p5, m6, h6, n6, p6, CAsn2


def CTX_BG_TH_network_CL(pd, corstim, tmax, dt, n, Idbs, Iappco, states = []):
    t = np.arange(0, tmax + dt, dt)
    
    if not states:
        v1_init = -62 + np.random.normal(loc=0, scale=5, size=(n, 1))
        v2_init = -62 + np.random.normal(loc=0, scale=5, size=(n, 1))
        v3_init = -62 + np.random.normal(loc=0, scale=5, size=(n, 1))
        v4_init = -62 + np.random.normal(loc=0, scale=5, size=(n, 1))
        v5_init = -63.8 + np.random.normal(loc=0, scale=5, size=(n, 1))
        v6_init = -63.8 + np.random.normal(loc=0, scale=5, size=(n, 1))
        vth_0, vsn_0, vge_0, vgi_0, vstr_indr_0, vstr_dr_0 = v1_init[:,0], v2_init[:,0], v3_init[:,0], v4_init[:,0], v5_init[:,0], v6_init[:,0]
        ve_0, vi_0 = -65.0 * np.ones(n), -65.0 * np.ones(n)
        ue_0, ui_0 = 0.2 * ve_0, 0.2 * vi_0
        N3_0, N4_0, H1_0, H3_0 = gpe_ninf(vge_0), gpe_ninf(vgi_0), th_hinf(vth_0), gpe_hinf(vge_0)
        H4_0, R1_0, R3_0, R4_0 = gpe_hinf(vgi_0), th_rinf(vth_0), gpe_rinf(vge_0), gpe_rinf(vgi_0)
        CA2_0 = 0.1 * np.ones(n); CA3_0 = 0.1 * np.ones(n); CA4_0 = 0.1 * np.ones(n)
        CAsn2_0 = 0.005 * np.ones(n)
        N2_0, H2_0, M2_0, A2_0 = stn_ninf(vsn_0), stn_hinf(vsn_0), stn_minf(vsn_0), stn_ainf(vsn_0)
        B2_0, C2_0, D2_0, D1_0 = stn_binf(vsn_0), stn_cinf(vsn_0), stn_d2inf(CAsn2_0), stn_d1inf(vsn_0)
        P2_0, Q2_0, R2_0 = stn_pinf(vsn_0), stn_qinf(vsn_0), stn_rinf(CAsn2_0)
        m5_0 = alpham(vstr_indr_0)/(alpham(vstr_indr_0)+betam(vstr_indr_0)); h5_0 = alphah(vstr_indr_0)/(alphah(vstr_indr_0)+betah(vstr_indr_0))
        n5_0 = alphan(vstr_indr_0)/(alphan(vstr_indr_0)+betan(vstr_indr_0)); p5_0 = alphap(vstr_indr_0)/(alphap(vstr_indr_0)+betap(vstr_indr_0))
        m6_0 = alpham(vstr_dr_0)/(alpham(vstr_dr_0)+betam(vstr_dr_0)); h6_0 = alphah(vstr_dr_0)/(alphah(vstr_dr_0)+betah(vstr_dr_0))
        n6_0 = alphan(vstr_dr_0)/(alphan(vstr_dr_0)+betan(vstr_dr_0)); p6_0 = alphap(vstr_dr_0)/(alphap(vstr_dr_0)+betap(vstr_dr_0))
        v1, v2, v3, v4, v5, v6 = v1_init, v2_init, v3_init, v4_init, v5_init, v6_init
    else:
        vth_0, vsn_0, vge_0, vgi_0 = states['V1'], states['V2'], states['V3'], states['V4']
        vstr_indr_0, vstr_dr_0, ve_0, vi_0 = states['V5'], states['V6'], states['V7'], states['V8']
        
        # Use .get() or slicing to handle shape differences between scalar states and matrix states
        def get_last(val, fallback):
            if val is None: return fallback
            if isinstance(val, (np.ndarray, list)) and np.ndim(val) > 1: return val[:, -1]
            return val
        
        ue_0 = get_last(states.get('ue'), 0.2 * ve_0)
        ui_0 = get_last(states.get('ui'), 0.2 * vi_0)
        
        N3_0, N4_0 = states['N3'], states['N4']
        H1_0, H3_0, H4_0 = states['H1'], states['H3'], states['H4']
        R1_0, R3_0, R4_0 = states['R1'], states['R3'], states['R4']
        CA2_0, CA3_0, CA4_0 = states['CA2'], states['CA3'], states['CA4']
        N2_0, H2_0, M2_0, A2_0 = states['N2'], states['H2'], states['M2'], states['A2']
        B2_0, C2_0, D2_0, D1_0 = states['B2'], states['C2'], states['D2'], states['D1']
        P2_0, Q2_0, R2_0 = states['P2'], states['Q2'], states['R2']
        m5_0, h5_0, n5_0, p5_0 = states['m5'], states['h5'], states['n5'], states['p5']
        m6_0, h6_0, n6_0, p6_0 = states['m6'], states['h6'], states['n6'], states['p6']
        CAsn2_0 = states.get('CAsn2', 0.005 * np.ones(n)) # fallback for original states
        v1, v2, v3, v4, v5, v6 = states['v1'], states['v2'], states['v3'], states['v4'], states['v5'], states['v6']

    all_idx, bll_idx, cll_idx, dll_idx, ell_idx, fll_idx = [np.random.permutation(n) for _ in range(6)]
    gll_idx, hll_idx, ill_idx, jll_idx, kll_idx, lll_idx = [np.random.permutation(n) for _ in range(6)]
    mll_idx, nll_idx, oll_idx = [np.random.permutation(n) for _ in range(3)]
    
    gcorsna = 0.3 * np.random.rand(n, 1)
    gcorsnn = 0.003 * np.random.rand(n, 1)
    gcordrstr = (0.07 - 0.044 * pd) + 0.001 * np.random.rand(n, 1)
    ggege = np.random.rand(n, 1)

    gsngen = np.zeros(n); gsngen[np.random.permutation(n)[:2]] = 0.002 * np.random.rand(2)
    gsngea = np.zeros(n); gsngea[np.random.permutation(n)[:2]] = 0.3 * np.random.rand(2)
    gsngi = np.zeros(n); gsngi[np.random.permutation(n)[:5]] = 0.15

    # Precomputed indices
    _n = np.arange(n)
    r_p1, r_p2 = (_n - 1) % n, (_n - 2) % n
    r_m1, r_m2, r_m3, r_m4, r_m5, r_m6, r_m7, r_m8, r_m9 = [(_n + i) % n for i in range(1, 10)]

    # Precompute syn masks
    t_a = 1000; t_vec = np.arange(0, t_a + dt, dt); gpeak, gpeak1, tau = 0.43, 0.3, 5
    const = gpeak / (tau * np.exp(-1)); const1 = gpeak1 / (tau * np.exp(-1)); const2 = gpeak1 / (tau * np.exp(-1))

    t_d_th_cor = 5; syn_func_th = const * (t_vec - t_d_th_cor) * (np.exp(-(t_vec - t_d_th_cor) / tau)) * ((t_vec >= t_d_th_cor) & (t_vec <= t_a))
    t_d_stn_gpe = 2; taudstngpea, taurstngpea, taudstngpen, taurstngpen = 2.5, 0.4, 67, 2
    tpkg_ea = t_d_stn_gpe + (((taudstngpea * taurstngpea) / (taudstngpea - taurstngpea)) * np.log(taudstngpea / taurstngpea))
    fsea = 1 / (np.exp(-(tpkg_ea - t_d_stn_gpe) / taudstngpea) - np.exp(-(tpkg_ea - t_d_stn_gpe) / taurstngpea))
    syn_func_stn_gpea = gpeak*fsea*(np.exp(-(t_vec-t_d_stn_gpe)/taudstngpea)-np.exp(-(t_vec-t_d_stn_gpe)/taurstngpea))*((t_vec>=t_d_stn_gpe)&(t_vec<=t_a))
    tpkg_en = t_d_stn_gpe + (((taudstngpen * taurstngpen) / (taudstngpen - taurstngpen)) * np.log(taudstngpen / taurstngpen))
    fsen = 1 / (np.exp(-(tpkg_en - t_d_stn_gpe) / taudstngpen) - np.exp(-(tpkg_en - t_d_stn_gpe) / taurstngpen))
    syn_func_stn_gpen = gpeak*fsen*(np.exp(-(t_vec-t_d_stn_gpe)/taudstngpen)-np.exp(-(t_vec-t_d_stn_gpe)/taurstngpea))*((t_vec>=t_d_stn_gpe)&(t_vec<=t_a))
    t_d_stn_gpi = 1.5; syn_func_stn_gpi = const*(t_vec-t_d_stn_gpi)*(np.exp(-(t_vec-t_d_stn_gpi)/tau))*((t_vec>=t_d_stn_gpi)&(t_vec<=t_a))
    t_d_gpe_stn = 4; taudg, taurg = 7.7, 0.4; tpkg_g = t_d_gpe_stn + (((taudg * taurg) / (taudg - taurg)) * np.log(taudg / taurg))
    fg = 1 / (np.exp(-(tpkg_g - t_d_gpe_stn) / taudg) - np.exp(-(tpkg_g - t_d_gpe_stn) / taurg))
    syn_func_gpe_stn = gpeak1*fg*(np.exp(-(t_vec-t_d_gpe_stn)/taudg)-np.exp(-(t_vec-t_d_gpe_stn)/taurg))*((t_vec>=t_d_gpe_stn)&(t_vec<=t_a))
    t_d_gpe_gpi = 3; syn_func_gpe_gpi = const1*(t_vec-t_d_gpe_gpi)*(np.exp(-(t_vec-t_d_gpe_gpi)/tau))*((t_vec>=t_d_gpe_gpi)&(t_vec<=t_a))
    t_d_gpe_gpe = 1; syn_func_gpe_gpe = const1*(t_vec-t_d_gpe_gpe)*(np.exp(-(t_vec-t_d_gpe_gpe)/tau))*((t_vec>=t_d_gpe_gpe)&(t_vec<=t_a))
    t_d_gpi_th = 5; syn_func_gpi_th = const1*(t_vec-t_d_gpi_th)*(np.exp(-(t_vec-t_d_gpi_th)/tau))*((t_vec>=t_d_gpi_th)&(t_vec<=t_a))
    t_d_d2_gpe = 5; syn_func_str_indr = const2*(t_vec-t_d_d2_gpe)*(np.exp(-(t_vec-t_d_d2_gpe)/tau))*((t_vec>=t_d_d2_gpe)&(t_vec<=t_a))
    t_d_d1_gpi = 4; syn_func_str_dr = const2*(t_vec-t_d_d1_gpi)*(np.exp(-(t_vec-t_d_d1_gpi)/tau))*((t_vec>=t_d_d1_gpi)&(t_vec<=t_a))
    t_d_cor_d2 = 5.1; syn_func_cor_d2 = const*(t_vec-t_d_cor_d2)*(np.exp(-(t_vec-t_d_cor_d2)/tau))*((t_vec>=t_d_cor_d2)&(t_vec<=t_a))
    t_d_cor_stn = 5.9; taudn, taurn, tauda, taura = 90, 2, 2.49, 0.5; tpka = t_d_cor_stn+(((tauda*taura)/(tauda-taura))*np.log(tauda/taura))
    fa = 1/(np.exp(-(tpka-t_d_cor_stn)/tauda)-np.exp(-(tpka-t_d_cor_stn)/taura))
    syn_func_cor_stn_a = gpeak*fa*(np.exp(-(t_vec-t_d_cor_stn)/tauda)-np.exp(-(t_vec-t_d_cor_stn)/taura))*((t_vec>=t_d_cor_stn)&(t_vec<=t_a))
    tpkn = t_d_cor_stn+(((taudn*taurn)/(taudn-taurn))*np.log(taudn/taurn)); fn = 1/(np.exp(-(tpkn-t_d_cor_stn)/taudn)-np.exp(-(tpkn-t_d_cor_stn)/taurn))
    syn_func_cor_stn_n = gpeak*fn*(np.exp(-(t_vec-t_d_cor_stn)/taudn)-np.exp(-(t_vec-t_d_cor_stn)/taurn))*((t_vec>=t_d_cor_stn)&(t_vec<=t_a))

    res = _CTX_BG_TH_network_CL_core(
        pd, corstim, tmax, dt, n, Idbs, Iappco,
        vth_0, vsn_0, vge_0, vgi_0, vstr_indr_0, vstr_dr_0, ve_0, ue_0, vi_0, ui_0,
        N3_0, N4_0, H1_0, H3_0, H4_0, R1_0, R3_0, R4_0, CA2_0, CA3_0, CA4_0,
        N2_0, H2_0, M2_0, A2_0, B2_0, C2_0, D2_0, D1_0, P2_0, Q2_0, R2_0,
        m5_0, h5_0, n5_0, p5_0, m6_0, h6_0, n6_0, p6_0,
        CAsn2_0,
        all_idx, bll_idx, cll_idx, dll_idx, ell_idx, fll_idx, gll_idx, hll_idx, ill_idx, jll_idx, kll_idx, lll_idx, mll_idx, nll_idx, oll_idx,
        gcorsna, gcorsnn, gcordrstr, ggege,
        gsngen, gsngea, gsngi,
        r_p1, r_p2, r_m1, r_m2, r_m3, r_m4, r_m5, r_m6, r_m7, r_m8, r_m9,
        syn_func_th, syn_func_stn_gpea, syn_func_stn_gpen, syn_func_stn_gpi, syn_func_gpe_stn, syn_func_gpe_gpi, syn_func_gpe_gpe, syn_func_gpi_th, syn_func_str_indr, syn_func_str_dr, syn_func_cor_d2, syn_func_cor_stn_a, syn_func_cor_stn_n
    )
    
    vth, vsn, vge, vgi, vstr_indr, vstr_dr, ve, vi, ue, ui, \
    N3, N4, H1, H3, H4, R1, R3, R4, CA2, CA3, CA4, \
    N2, H2, M2, A2, B2, C2, D2, D1, P2, Q2, R2, \
    m5, h5, n5, p5, m6, h6, n6, p6, CAsn2 = res

    TH_APs = find_spike_times(vth, t, n)
    STN_APs = find_spike_times(vsn, t, n)
    GPe_APs = find_spike_times(vge, t, n)
    GPi_APs = find_spike_times(vgi, t, n)
    Striat_APs_indr = find_spike_times(vstr_indr, t, n)
    Striat_APs_dr = find_spike_times(vstr_dr, t, n)
    Cor_APs = find_spike_times(np.concatenate((ve, vi), axis=0), t, 2 * n)

    states = {'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4, 'v5': v5, 'v6': v6,
              'vth': vth, 'vsn': vsn, 'vge': vge, 'vgi': vgi, 'vstr_indr': vstr_indr, 'vstr_dr': vstr_dr, 've': ve, 'vi': vi,
              'V1': vth[:, -1], 'V2': vsn[:, -1], 'V3': vge[:, -1], 'V4': vgi[:, -1], 'V5': vstr_indr[:, -1], 'V6': vstr_dr[:, -1], 'V7': ve[:, -1], 'V8': vi[:, -1],
              'ue': ue, 'ui': ui,
              'N3': N3, 'N4': N4, 'H1': H1, 'H3': H3, 'H4': H4, 'R1': R1, 'R3': R3, 'R4': R4, 'CA2': CA2, 'CA3': CA3, 'CA4': CA4,
              'N2': N2, 'H2': H2, 'M2': M2, 'A2': A2, 'B2': B2, 'C2': C2, 'D2': D2, 'D1': D1, 'P2': P2, 'Q2': Q2, 'R2': R2,
              'm5': m5, 'h5': h5, 'n5': n5, 'p5': p5, 'm6': m6, 'h6': h6, 'n6': n6, 'p6': p6, 'CAsn2': CAsn2}

    return TH_APs, STN_APs, GPe_APs, GPi_APs, Striat_APs_indr, Striat_APs_dr, Cor_APs, states

def creatdbs(pattern, tmax, dt, PW, amplitude):
    t = np.arange(0, tmax + dt, dt)
    Idbs = np.zeros_like(t)
    iD = amplitude
    pulse = iD * np.ones(int(PW/dt))

    i = 0
    while i < len(t)-1:
        pulse_len = int(PW/dt)
        if (i + pulse_len) > len(Idbs):
            pulse_len = len(Idbs) - i
        Idbs[i:i+pulse_len] = pulse[:pulse_len]
        instfreq = pattern
        isi = 1000 / instfreq
        i += round(isi / dt)

    return Idbs

# @jit(nopython=False) 
def find_spike_times(voltage, time, num_neurons):
    data = []
    time = time / 1000  # unit: second
    for k in range(num_neurons):
        spike_times = time[np.diff((voltage[k, :] > -20).astype(int), prepend=0) == 1]
        data.append({'times': spike_times.tolist()})
    return data

@jit(nopython=True)
def gpe_ainf(V):
    return 1 / (1 + np.exp(-(V + 57) / 2))

@jit(nopython=True)
def gpe_hinf(V):
    return 1 / (1 + np.exp((V + 58) / 12))

@jit(nopython=True)
def gpe_minf(V):
    return 1 / (1 + np.exp(-(V + 37) / 10))

@jit(nopython=True)
def gpe_ninf(V):
    return 1 / (1 + np.exp(-(V + 50) / 14))

@jit(nopython=True)
def gpe_rinf(V):
    return 1 / (1 + np.exp((V + 70) / 2))


@jit(nopython=True)
def gpe_sinf(V):
    return 1 / (1 + np.exp(-(V + 35) / 2))

@jit(nopython=True)
def gpe_tauh(V):
    return 0.05 + 0.27 / (1 + np.exp(-(V + 40) / -12))

@jit(nopython=True) 
def gpe_taun(V):
    return 0.05 + 0.27 / (1 + np.exp(-(V + 40) / -12))

@jit(nopython=True) 
def th_hinf(V):
    return 1 / (1 + np.exp((V + 41) / 4))

@jit(nopython=True) 
def th_minf(V):
    return 1 / (1 + np.exp(-(V + 37) / 7))

@jit(nopython=True) 
def th_pinf(V):
    return 1 / (1 + np.exp(-(V + 60) / 6.2))

@jit(nopython=True) 
def th_rinf(V):
    return 1 / (1 + np.exp((V + 84) / 4))

@jit(nopython=True) 
def th_tauh(V):
    return 1 / (ah(V) + bh(V))

@jit(nopython=True) 
def ah(V):
    return 0.128 * np.exp(-(V + 46) / 18)

@jit(nopython=True) 
def bh(V):
    return 4 / (1 + np.exp(-(V + 23) / 5))

@jit(nopython=True) 
def th_taur(V):
    return 0.15 * (28 + np.exp(-(V + 25) / 10.5))

@jit(nopython=True) 
def alphah(V):
    return 0.128 * np.exp((-50 - V) / 18)

@jit(nopython=True) 
def alpham(V):
    return (0.32 * (54 + V)) / (1 - np.exp((-54 - V) / 4))

@jit(nopython=True) 
def alphan(V):
    return (0.032 * (52 + V)) / (1 - np.exp((-52 - V) / 5))

@jit(nopython=True) 
def alphap(V):
    return (3.209e-4 * (30 + V)) / (1 - np.exp((-30 - V) / 9))

@jit(nopython=True) 
def betah(V):
    return 4 / (1 + np.exp((-27 - V) / 5))

@jit(nopython=True) 
def betan(V):
    return 0.5 * np.exp((-57 - V) / 40)

@jit(nopython=True) 
def betam(V):
    return 0.28 * (V + 27) / (np.exp((27 + V) / 5) - 1)

@jit(nopython=True) 
def betap(V):
    return (-3.209e-4 * (30 + V)) / (1 - np.exp((30 + V) / 9))

@jit(nopython=True) 
def Ggaba(V):
    return 2 * (1 + np.tanh(V / 4))

@jit(nopython=True) 
def stn_ainf(V):
    return 1 / (1 + np.exp(-(V + 45) / 14.7))

@jit(nopython=True) 
def stn_binf(V):
    return 1 / (1 + np.exp((V + 90) / 7.5))

@jit(nopython=True) 
def stn_cinf(V):
    return 1 / (1 + np.exp(-(V + 30.6) / 5))

@jit(nopython=True) 
def stn_d1inf(V):
    return 1 / (1 + np.exp((V + 60) / 7.5))

@jit(nopython=True) 
def stn_d2inf(V):
    return 1 / (1 + np.exp((V - 0.1) / 0.02))
    # replace with scipy.special.expit to fix overflow warning
    # return expit(-(V-0.1)/0.02)

@jit(nopython=True) 
def stn_hinf(V):
    return 1 / (1 + np.exp((V + 45.5) / 6.4))

@jit(nopython=True) 
def stn_minf(V):
    return 1 / (1 + np.exp(-(V + 40) / 8))

@jit(nopython=True) 
def stn_ninf(V):
    return 1 / (1 + np.exp(-(V + 41) / 14))

@jit(nopython=True) 
def stn_pinf(V):
    return 1 / (1 + np.exp(-(V + 56) / 6.7))

@jit(nopython=True) 
def stn_qinf(V):
    return 1 / (1 + np.exp((V + 85) / 5.8))

@jit(nopython=True) 
def stn_rinf(V):
    return 1 / (1 + np.exp(-(V - 0.17) / 0.08))
    # replace with scipy.special.expit to fix overflow warning
    # return expit((V - 0.17)/0.08)

@jit(nopython=True) 
def stn_taua(V):
    return 1 + 1 / (1 + np.exp(-(V + 40) / -0.5))

@jit(nopython=True) 
def stn_taub(V):
    return 200 / (np.exp(-(V + 60) / -30) + np.exp(-(V + 40) / 10))

@jit(nopython=True) 
def stn_tauc(V):
    return 45 + 10 / (np.exp(-(V + 27) / -20) + np.exp(-(V + 50) / 15))

@jit(nopython=True) 
def stn_taud1(V):
    return 400 + 500 / (np.exp(-(V + 40) / -15) + np.exp(-(V + 20) / 20))

@jit(nopython=True) 
def stn_tauh(V):
    return 24.5 / (np.exp(-(V + 50) / -15) + np.exp(-(V + 50) / 16))

@jit(nopython=True) 
def stn_taum(V):
    return 0.2 + 3 / (1 + np.exp(-(V + 53) / -0.7))

@jit(nopython=True) 
def stn_taun(V):
    return 11 / (np.exp(-(V + 40) / -40) + np.exp(-(V + 40) / 50))

@jit(nopython=True) 
def stn_taup(V):
    return 5 + 0.33 / (np.exp(-(V + 27) / -10) + np.exp(-(V + 102) / 15))

@jit(nopython=True) 
def stn_tauq(V):
    return 400 / (np.exp(-(V + 50) / -15) + np.exp(-(V + 50) / 16))

def make_Spectrum(raw, params):
    # Compute Multitaper Spectrum

    # Quick check if raw has any spikes to avoid empty array math/NaNs
    has_spikes = False
    for d in raw:
        if len(d['times']) > 0:
            has_spikes = True
            break
            
    if not has_spikes:
        return 0.0, np.array([]), np.array([])

    try:
        S, f = mtspectrumpt(raw, params)
    except Exception as e:
        # Gracefully handle dpss failure when spikes are too sparse/synchronized
        return 0.0, np.array([]), np.array([])

    # Extract beta frequencies
    beta = S[(f > 7) & (f < 35)]
    betaf = f[(f > 7) & (f < 35)]

    # Compute area under the beta spectrum
    area = np.trapz(beta, betaf)

    return area, S, f

def mtspectrumpt(data, params=None, fscorr=0, t=None):
    if data is None:
        raise ValueError('Need data')
    
    if params is None:
        params = {}

    if 'tapers' not in params:
        params['tapers'] = None

    if 'pad' not in params:
        params['pad'] = 0

    if 'Fs' not in params:
        params['Fs'] = 1.0

    if 'fpass' not in params:
        params['fpass'] = [0, 100]

    if 'err' not in params:
        params['err'] = [1, 0]

    if 'trialave' not in params:
        params['trialave'] = False

    if 'params' in params:
        del params['params']
    
    
    data = change_row_to_column(data)

    if not t:
        mintime, maxtime = minmaxsptimes(data)
        dt = 1 / params['Fs']  # sampling time
        t = np.arange(mintime - dt, maxtime + 2 * dt, dt)  # time grid for prolates

    N = len(t)  # number of points in grid for dpss
    nfft = max(2 ** (np.ceil(np.log2(N)) + params['pad']), N)  # number of points in fft of prolates
    nfft = int(nfft)

    f, findx = getfgrid(params['Fs'], nfft, params['fpass'])  # get frequency grid for evaluation

    tapers, eigs = dpsschk(params['tapers'], N, params['Fs'])  # check tapers

    J, Msp, Nsp = mtfftpt(data, tapers, nfft, t, f, findx)  # mt fft for point process times

    # Average over tapers
    S = np.mean(np.real(np.conj(J) * J), axis=1) # Result shape: (nfreq, C)
    
    if params['trialave']:
        # Average over channels (neurons)
        if S.ndim > 1:
            S = np.mean(S, axis=1) # Result shape: (nfreq,)
        Msp = np.mean(Msp)
    else:
        # If not trial averaging, but only 1 channel, still return 1D
        if S.ndim > 1 and S.shape[1] == 1:
            S = S[:, 0] # Result shape: (nfreq,)

    R = Msp * params['Fs']

    if params['err'][0] == 0:
        raise ValueError('Cannot compute error bars with err[0] = 0; change params and run again')

    return S, f

def getparams(params):
    if 'tapers' not in params or not params['tapers']:
        print('Tapers unspecified, defaulting to params.tapers=[3, 5]')
        params['tapers'] = [3, 5]

    if params and len(params['tapers']) == 3:
        # Compute time-bandwidth product
        TW = params['tapers'][1] * params['tapers'][0]
        # Compute number of tapers
        K = int(np.floor(2 * TW - params['tapers'][2]))
        params['tapers'] = [TW, K]

    if 'pad' not in params or not params['pad']:
        params['pad'] = 0

    if 'Fs' not in params or not params['Fs']:
        params['Fs'] = 1

    if 'fpass' not in params or not params['fpass']:
        params['fpass'] = [0, params['Fs'] / 2]

    if 'err' not in params or not params['err']:
        params['err'] = 0

    if 'trialave' not in params or not params['trialave']:
        params['trialave'] = 0

    tapers = params['tapers']
    pad = params['pad']
    Fs = params['Fs']
    fpass = params['fpass']
    err = params['err']
    trialave = params['trialave']

    return tapers, pad, Fs, fpass, err, trialave, params

def change_row_to_column(data):
    dtmp = []
    if isinstance(data, dict):
        C = len(data)
        if C == 1:
            dtmp = data[list(data.keys())[0]]
            data = dtmp[:]
    else:
        N, C = data.shape if hasattr(data, 'shape') else (len(data), 1)
        if N == 1 or C == 1:
            data = np.array(data).ravel()
    return data


def minmaxsptimes(data):
    dtmp = ""
    if isinstance(data, np.ndarray):
        values = [value for d in data for value in d.values()]
        mintime = [min(d) if d else float('nan') for d in values]
        maxtime = [max(d) if d else float('nan') for d in values]
        mintime = np.nanmin(mintime)
        maxtime = np.nanmax(maxtime)


    else:
        print('in minmaxsptimes, np.ndarray type was not found!')
        dtmp = data
        if np.any(dtmp) and dtmp.size > 0:
            maxtime = max(dtmp)
            mintime = min(dtmp)
        else:
            mintime = float('nan')
            maxtime = float('nan')

    if mintime < 0:
        raise ValueError('Minimum spike time is negative')
    
    return mintime, maxtime


def getfgrid(Fs, nfft, fpass):
    if len(fpass) != 1:
        df = Fs / nfft
        f = np.arange(0, Fs, df)[:nfft]
        findx = np.where((f >= fpass[0]) & (f <= fpass[-1]))[0]
    else:
        fmin = np.abs(np.arange(0, Fs, Fs / nfft)[:nfft] - fpass[0])
        findx = np.argmin(fmin)

    f = f[findx]
    return f, findx


def dpsschk(tapers, N, Fs):
    if not (hasattr(tapers, "__len__") and len(tapers) == 2):
        raise ValueError("Tapers must be a list or array of two values [TW, K]")
        
    if len(tapers) == 2:
        tapers, eigs = dpss(N, tapers[0], tapers[1], return_ratios=True)
        tapers *= np.sqrt(Fs)
    elif N != len(tapers):
        raise ValueError("Number of time points is different from the length of the tapers")
    
    return tapers, eigs

def mtfftpt(data, tapers, nfft, t, f, findx):

    if type(data) is np.ndarray:
        C = len(data)
    else:
        C = 1
        
    # K = tapers.shape[1]  # number of tapers
    K = len(tapers)

    nfreq = len(f)  # number of frequencies
    if nfreq != len(findx):
        raise ValueError('Frequency information (last two arguments) inconsistent')

    H = np.fft.fft(tapers.T, nfft, axis=0)  # fft of tapers

    H = H[findx, :]  # restrict fft of tapers to required frequencies
    w = 2 * np.pi * f  # angular frequencies at which ft is to be evaluated

    Nsp = np.zeros(C)
    Msp = np.zeros(C)
    J = np.zeros((nfreq, K, C), dtype=complex)

    for ch in range(C):

        if isinstance(data, np.ndarray):
            dtmp = data[ch]['times']
            indx = np.where((dtmp >= min(t)) & (dtmp <= max(t)))[0]

        else:
            dtmp = data
            indx = np.where((dtmp >= min(t)) & (dtmp <= max(t)))[0]
            if indx.size > 0:
                dtmp = dtmp[indx]
                       

        Nsp[ch] = len(dtmp)
        Msp[ch] = Nsp[ch] / len(t)

        if Msp[ch] != 0:
            data_proj = interp1d(t, tapers, kind='linear', fill_value=0, bounds_error=False)(dtmp)
            exponential = np.exp(-1j * np.outer(w, (dtmp - t[0])))
            J[:, :, ch] = exponential @ data_proj.T - H * Msp[ch] # changed @ since Msp[ch] is a type double
        else:
            J[:, :, ch] = 0

    return J, Msp, Nsp

def specerr(S, J, err, trialave, numsp=None):
    if numsp is None:
        numsp = np.ones(J.shape[2], dtype=int)

    nf, K, C = J.shape
    errchk = err[0]
    p = err[1]
    pp = 1 - p / 2
    qq = 1 - pp

    if trialave:
        dim = K * C
        C = 1
        dof = 2 * dim
        if numsp is not None:
            dof = int(1 / (1 / dof + 1 / (2 * np.sum(numsp))))

        J = J.reshape(nf, dim)
    else:
        dim = K
        dof = 2 * dim * np.ones(C)
        if numsp is not None:
            for ch in range(C):
                dof[ch] = int(1 / (1 / dof[ch] + 1 / (2 * numsp[ch])))

    Serr = np.zeros((2, nf, C))

    if errchk == 1:
        Qp = chi2.ppf(pp, dof)
        Qq = chi2.ppf(qq, dof)

        Serr[0, :, :] = (dof * S / Qp).reshape(len(dof * S / Qp), -1) # modified to from (N,) -> (N,1)
        Serr[1, :, :] = (dof * S / Qq).reshape(len(dof * S / Qq), -1)
        
        # Serr[0, :, :] = np.reshape(len(dof * S / Qp), -1) 
        # Serr[1, :, :] = np.reshape(len(dof * S / Qq), -1)

        # Serr[0, :, :] = dof * S / Qp
        # Serr[1, :, :] = dof * S / Qq
    elif errchk == 2:
        tcrit = t.ppf(pp, dim - 1)
        Sjk = np.zeros((dim, nf, C))
        for k in range(dim):
            indices = np.setdiff1d(np.arange(dim), k)
            Jjk = J[:, indices, :]  # 1-drop projection
            eJjk = np.sum(Jjk * np.conj(Jjk), axis=1)
            Sjk[k, :, :] = eJjk / (dim - 1)  # 1-drop spectrum

        sigma = np.sqrt(dim - 1) * np.std(np.log(Sjk), axis=0)
        conf = np.tile(tcrit, (nf, C)) * sigma
        Serr[0, :, :] = S * np.exp(-conf)
        Serr[1, :, :] = S * np.exp(conf)

    return np.squeeze(Serr)
