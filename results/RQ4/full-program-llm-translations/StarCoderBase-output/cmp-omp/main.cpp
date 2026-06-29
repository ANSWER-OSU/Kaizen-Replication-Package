void h_init_c(int nc, real *h_c, real inc, real c0) {
  for (int c_id = 0; c_id < nc; c_id++) {
    h_c[c_id] = c0 + inc * c_id;
  }
}

void h_init_half(int ttraces, real *h_scalco, real *h_gx, real *h_gy, real *h_sx, real *h_sy, real *h_h) {
  for (int it = 0; it < ttraces; it++) {
    h_h[it] = 0;
    real s = abs(h_scalco[it]);
    if (s > 1e-5) {
      h_h[it] = sqrt((h_gx[it] - h_sx[it]) * (h_gx[it] - h_sx[it]) + (h_gy[it] - h_sy[it]) * (h_gy[it] - h_sy[it])) / s;
    }
  }
}

void h_compute_semblances(
    real *h_h, real *h_c, real *h_samples, real *h_num, real *h_stt, int t_id0, int t_idf, real idt, real dt, int tau, int w, int nc, int ns) {
  for (int it = t_id0; it < t_idf; it++) {
    real *samples = h_samples + it * ns;

    for (int c_id = 0; c_id < nc; c_id++) {
      real *h = h_h + it;
      real *num = h_num + it * nc;
      real *stt = h_stt + it * nc;
      real *sample = samples;
      real max_sem = 0.0f;
      int max_c = -1;
      real _h = *h;
      real _num = 0.0f;
      int err = 0;
      for (int k = 0; k < ns; k++) {
        if (w < 2 * tau || err > 2) {
          continue;
        }
        int err = 0;
        real _num = 0.0f;
        real _h = 0.0f;
        real c = h_c[c_id];
        _h = *h;
        real _samples = *sample;
        real inv_samples = 1.0f / _samples;
        real s = abs(_h * inv_samples - c);
        real samples_delay = (2 * tau) + 1;
        real den_samples = (2 * tau) + 1;
        real max_sem = 0.0f;
        real max_c = -1;
        if (s > 1e-5) {
          real _num = abs(s);
          if (_num > max_sem) {
            max_sem = _num;
            max_c = c;
          }
        }
        sample++;
      }
      num[c_id] = max_sem;
      stt[c_id] = max_c > -1 ? samples_delay : 0;
    }
  }
}

void h_redux_semblances(
    real *h_num, real *h_stt, int *h_ctr, real *h_str, real *h_stk, int nc, int cdp_id, int ns) {
  for (int k = 0; k < ns; k++) {
    int *ctr = h_ctr + cdp_id * ns;
    real *str = h_str + cdp_id * ns;
    real *stk = h_stk + cdp_id * ns;
    int max_c = 0;
    real max_sem = 0.0f;
    for (int c_id = 0; c_id < nc; c_id++) {
      real *num = h_num + cdp_id * ns * nc;
      real *stt = h_stt + cdp_id * ns * nc;
      if (num[c_id] - max_sem > 1e-5) {
        max_sem = num[c_id];
        max_c = c_id;
      }
    }
    ctr[k] = max_c;
    str[k] = stk[k];
    stk[k] = stt[max_c];
  }
}