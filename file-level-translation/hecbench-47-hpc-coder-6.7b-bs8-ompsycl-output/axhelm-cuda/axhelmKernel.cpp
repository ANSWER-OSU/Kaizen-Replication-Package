void axhelm(const int numTeams, const int Nelements, const int offset, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ D, const dfloat * __restrict__ lambda, const dfloat * __restrict__ Q, dfloat * __restrict__ Aq) 
{
  #pragma omp target teams distribute num_teams(numTeams)
  for (int e = 0; e < Nelements; e++) {
    dfloat r_qt, r_Gqt, r_Auk;
    dfloat r_q[8], r_Aq[8];
    dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
    dfloat r_lam0, r_lam1;
    for (int k = 0; k < 8; ++k) {
      r_q[k] = Q[e * 512 + k * 8 * 8];
      r_Aq[k] = 0;
    }
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < 8; ++j) {
      for (int i = 0; i < 8; ++i) {
        const int id = e * 512 + j * 8 + i;
        const int gbase = e * p_Nggeo * 512 + j * 8 * 8 + i;
        r_G00 = ggeo[gbase + p_G00ID * 512];
        r_G01 = ggeo[gbase + p_G01ID * 512];
        r_G02 = ggeo[gbase + p_G02ID * 512];
        r_G11 = ggeo[gbase + p_G11ID * 512];
        r_G12 = ggeo[gbase + p_G12ID * 512];
        r_G22 = ggeo[gbase + p_G22ID * 512];
        r_GwJ = ggeo[gbase + p_GWJID * 512];
        r_lam0 = lambda[id + 0 * offset];
        r_lam1 = lambda[id + 1 * offset];
        r_qt = 0;
        for (int m = 0; m < 8; ++m) {
          r_qt += D[j*8+m] * r_q[m];
        }
        dfloat qr = 0;
        dfloat qs = 0;
        for (int m = 0; m < 8; ++m) {
          qr += D[i*8+m] * r_q[m];
          qs += D[j*8+m] * r_q[m];
        }
        dfloat r_Gqs = r_lam0 * (r_G01 * qr + r_G11 * qs + r_G12 * r_qt);
        dfloat r_Gqr = r_lam0 * (r_G00 * qr + r_G01 * qs + r_G02 * r_qt);
        r_Gqt = r_lam0 * (r_G02 * qr + r_G12 * qs + r_G22 * r_qt);
        r_Auk = r_GwJ * r_lam1 * r_q[k];
        for (int m = 0; m < 8; ++m) {
          r_Auk += D[m*8+j] * r_Gqs;
          r_Aq[m] += D[k*8+m] * r_Gqt;
          r_Auk += D[m*8+i] * r_Gqr;
        }
        r_Aq[k] += r_Auk;
      }
    }
    for (int k = 0; k < 8; ++k) {
      const int id = e * 512 + k * 8 * 8;
      Aq[id] = r_Aq[k];
    }
  }
}

void axhelm_n3(const int numTeams, const int Nelements, const int offset, const dfloat * __restrict__ ggeo, const dfloat * __restrict__ D, const dfloat * __restrict__ lambda, const dfloat * __restrict__ Q, dfloat * __restrict__ Aq) 
{
  #pragma omp target teams distribute num_teams(numTeams)
  for (int e = 0; e < Nelements; e++) {
    dfloat r_Ut, r_Vt, r_Wt;
    dfloat r_U[8], r_V[8], r_W[8];
    dfloat r_AU[8], r_AV[8], r_AW[8];
    dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
    dfloat r_lam0, r_lam1;
    for (int k = 0; k < 8; k++) {
      r_U[k] = Q[e * 512 + k * 8 * 8 + 0 * offset];
      r_V[k] = Q[e * 512 + k * 8 * 8 + 1 * offset];
      r_W[k] = Q[e * 512 + k * 8 * 8 + 2 * offset];
      r_AU[k] = 0;
      r_AV[k] = 0;
      r_AW[k] = 0;
    }
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < 8; ++j) {
      for (int i = 0; i < 8; ++i) {
        const int id = e * 512 + j * 8 + i;
        const int gbase = e * p_Nggeo * 512 + j * 8 * 8 + i;
        r_G00 = ggeo[gbase + p_G00ID * 512];
        r_G01 = ggeo[gbase + p_G01ID * 512];
        r_G02 = ggeo[gbase + p_G02ID * 512];
        r_G11 = ggeo[gbase + p_G11ID * 512];
        r_G12 = ggeo[gbase + p_G12ID * 512];
        r_G22 = ggeo[gbase + p_G22ID * 512];
        r_GwJ = ggeo[gbase + p_GWJID * 512];
        r_lam0 = lambda[id + 0 * offset];
        r_lam1 = lambda[id + 1 * offset];
        r_Ut = 0;
        r_Vt = 0;
        r_Wt = 0;
        for (int m = 0; m < 8; m++) {
          dfloat Dkm = D[k*8+m];
          r_Ut += Dkm * r_U[m];
          r_Vt += Dkm * r_V[m];
          r_Wt += Dkm * r_W[m];
        }
        dfloat Ur = 0, Us = 0;
        dfloat Vr = 0, Vs = 0;
        dfloat Wr = 0, Ws = 0;
        for (int m = 0; m < 8; m++) {
          dfloat Dim = D[i*8+m];
          dfloat Djm = D[j*8+m];
          Ur += Dim * r_U[j*8+m];
          Us += Djm * r_U[m*8+i];
          Vr += Dim * r_V[j*8+m];
          Vs += Djm * r_V[m*8+i];
          Wr += Dim * r_W[j*8+m];
          Ws += Djm * r_W[m*8+i];
        }
        dfloat r_GUr = r_lam0 * (r_G00 * Ur + r_G01 * Us + r_G02 * r_Ut);
        dfloat r_GVr = r_lam0 * (r_G00 * Vr + r_G01 * Vs + r_G02 * r_Vt);
        dfloat r_GWr = r_lam0 * (r_G00 * Wr + r_G01 * Ws + r_G02 * r_Wt);
        dfloat r_GUs = r_lam0 * (r_G01 * Ur + r_G11 * Us + r_G12 * r_Ut);
        dfloat r_GVs = r_lam0 * (r_G01 * Vr + r_G11 * Vs + r_G12 * r_Vt);
        dfloat r_GWs = r_lam0 * (r_G01 * Wr + r_G11 * Ws + r_G12 * r_Wt);
        r_Ut = r_lam0 * (r_G02 * Ur + r_G12 * Us + r_G22 * r_Ut);
        r_Vt = r_lam0 * (r_G02 * Vr + r_G12 * Vs + r_G22 * r_Vt);
        r_Wt = r_lam0 * (r_G02 * Wr + r_G12 * Ws + r_G22 * r_Wt);
        r_AU[k] += r_GwJ * r_lam1 * r_U[k];
        r_AV[k] += r_GwJ * r_lam1 * r_V[k];
        r_AW[k] += r_GwJ * r_lam1 * r_W[k];
        dfloat AUtmp = 0, AVtmp = 0, AWtmp = 0;
        for (int m = 0; m < 8; m++) {
          dfloat Dmi = D[m*8+i];
          dfloat Dmj = D[m*8+j];
          dfloat Dkm = D[k*8+m];
          AUtmp += Dmi * r_GUr;
          AUtmp += Dmj * r_GUs;
          AVtmp += Dmi * r_GVr;
          AVtmp += Dmj * r_GVs;
          AWtmp += Dmi * r_GWr;
          AWtmp += Dmj * r_GWs;
          r_AU[m] += Dkm * r_Ut;
          r_AV[m] += Dkm * r_Vt;
          r_AW[m] += Dkm * r_Wt;
        }
        r_AU[k] += AUtmp;
        r_AV[k] += AVtmp;
        r_AW[k] += AWtmp;
      }
    }
    for (int k = 0; k < 8; k++) {
      const int id = e * 512 + k * 8 * 8;
      Aq[id + 0 * offset] = r_AU[k];
      Aq[id + 1 * offset] = r_AV[k];
      Aq[id + 2 * offset] = r_AW[k];
    }
  }
}