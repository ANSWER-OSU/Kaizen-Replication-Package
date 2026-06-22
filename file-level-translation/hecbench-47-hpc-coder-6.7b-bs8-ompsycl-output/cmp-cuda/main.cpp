////////////////////////////////////////////////////////////////////////////////
/**
 * @file main.cpp
 * @date 2017-03-04
 * @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
////////////////////////////////////////////////////////////////////////////////


#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <omp.h>

#include "log.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "su_gather.hpp"

void
init_c(real *c, real inc, real c0, int nc) 
{
#pragma omp parallel for num_threads(nc)
  for (int i = 0; i < nc; i++) {
    c[i] = c0 + inc*i;
  }
}

void
init_half(const real* __restrict__ scalco, 
          const real* __restrict__ gx, 
          const real* __restrict__ gy, 
          const real* __restrict__ sx, 
          const real* __restrict__ sy, 
          real* __restrict__ h, 
          int ttraces) 
{
#pragma omp parallel for num_threads(ttraces)
  for (int i = 0; i < ttraces; i++) {
    real _s = scalco[i];

    if(-EPSILON < _s && _s < EPSILON) _s = 1.0f;
    else if(_s < 0) _s = 1.0f / _s;

    real hx = (gx[i] - sx[i]) * _s;
    real hy = (gy[i] - sy[i]) * _s;

    h[i] = 0.25f * (hx * hx + hy * hy) / FACTOR;
  }
}

void
compute_semblances(const real* __restrict__ h, 
                   const real* __restrict__ c, 
                   const real* __restrict__ samples, 
                   real* __restrict__ num,
                   real* __restrict__ stt,
                   int t_id0, 
                   int t_idf,
                   real idt,
                   real dt,
                   int tau,
                   int w,
                   int nc,
                   int ns,
                   int offset) 
{
  real den, ac_linear, ac_squared;
  real num_local[MAX_W],  m;
  int err;

  for(int i = offset; i < offset + ns * nc; i += omp_get_num_threads()) {
    real _c = c[i % nc];
    real _t0 = (dt * (i / nc)) * (i / nc);
    _t0 *= _t0;

    for(int j=0; j < w; j++) num_local[j] = 0.0f;

    den = 0.0f;
    ac_linear = 0.0f;
    ac_squared = 0.0f;
    m = 0.0f;
    err = 0;

    for(int t_id=t_id0; t_id < t_idf; t_id++) {
      real t = sqrtf(_t0 + _c * h[t_id]) * idt;

      int it = (int)( t );
      int ittau = it - tau;
      real x = t - (real)it;

      if(ittau >= 0 && it + tau + 1 < ns) {
        int k1 = ittau + (t_id-t_id0)*ns;
        real sk1p1 = samples[k1], sk1;

        for(int j=0; j < w; j++) {
          k1++;
          sk1 = sk1p1;
          sk1p1 = samples[k1];
          // linear interpolation optmized for this problema
          real v = (sk1p1 - sk1) * x + sk1;

          num_local[j] += v;
          den += v * v;
          ac_linear += v;
        }
        m += 1;
      } else { err++; }
    }

    // Reduction for num
    for(int j=0; j < w; j++) ac_squared += num_local[j] * num_local[j];

    // Evaluate semblances
    if(den > EPSILON && m > EPSILON && w > EPSILON && err < 2) {
      num[i] = ac_squared / (den * m);
      stt[i] = ac_linear  / (w   * m);
    }
    else {
      num[i] = -1.0f;
      stt[i] = -1.0f;
    }
  }
}

void
redux_semblances(const real* __restrict__ num, 
                 const real* __restrict__ stt, 
                 int*  __restrict__ ctr, 
                 real* __restrict__ str, 
                 real* __restrict__ stk,
                 const int nc, 
                 const int cdp_id,
                 const int ns,
                 const int offset) 
{
  for(int t0 = offset; t0 < offset + ns; t0++)
  {
    real max_sem = 0.0f;
    int max_c = -1;

    for(int it=t0*nc; it < (t0+1)*nc ; it++) {
      real _num = num[it];
      if(_num > max_sem) {
        max_sem = _num;
        max_c = it;
      }
    }

    ctr[cdp_id*ns + t0] = max_c % nc;
    str[cdp_id*ns + t0] = max_sem;
    stk[cdp_id*ns + t0] = max_c > -1 ? stt[max_c] : 0;
  }
}

int main(int argc, const char** argv) {
#ifdef SAVE
  std::ofstream c_out("cmp.c.su", std::ofstream::out | std::ios::binary);
  std::ofstream s_out("cmp.coher.su", std::ofstream::out | std::ios::binary);
  std::ofstream stack("cmp.stack.su", std::ofstream::out | std::ios::binary);
#endif

  // Parse command line and read arguments
  parser::add_argument("-c0", "C0 constant");
  parser::add_argument("-c1", "C1 constant");
  parser::add_argument("-nc", "NC constant");
  parser::add_argument("-aph", "APH constant");
  parser::add_argument("-tau", "Tau constant");
  parser::add_argument("-i", "Data path");
  parser::add_argument("-v", "Verbosity Level 0-3");

  parser::parse(argc, argv);

  // Read parameters and input
  const real c0 = std::stof(parser::get("-c0", true)) * FACTOR;
  const real c1 = std::stof(parser::get("-c1", true)) * FACTOR;
  const real itau = std::stof(parser::get("-tau", true));
  const int nc = std::stoi(parser::get("-nc", true));
  const int aph = std::stoi(parser::get("-aph", true));
  std::string path = parser::get("-i", true);
  logger::verbosity_level(std::stoi(parser::get("-v", false)));

  // Reads *.su data and starts gather
  su_gather gather(path, aph, nc);

  real *h_gx, *h_gy, *h_sx, *h_sy, *h_scalco, *h_samples, *h_str, *h_stk, dt;
  int *ntraces_by_cdp_id, *h_ctr;

  // Linearize gather data in order to improve data coalescence in GPU
  gather.linearize(ntraces_by_cdp_id, h_samples, dt, h_gx, h_gy, h_sx, h_sy, h_scalco, nc);
  const int  ttraces = gather.ttraces(); // Total traces -> Total amount of traces read
  const int  ncdps = gather().size();    // Number of cdps -> Total number of cdps read
  const int  ns = gather.ns();           // Number of samples
  const int  ntrs = gather.ntrs();       // Max number of traces by cdp
  const real inc = (c1-c0) * (1.0f / (real)nc);

  dt = dt / 1000000.0f;
  real idt = 1.0f / dt;
  int tau = ((int)( itau * idt) > 0) ? ((int)( itau * idt)) : 0;
  int w = (2 * tau) + 1;

  int number_of_semblances = 0;

  LOG(INFO, "Starting CMP execution");

  // Alloc memory
  real *gx, *gy, *sx, *sy, *scalco, *cdpsmpl;
  real *c, *num, *stt, *str, *stk; // nc stts per sample
  int  *ctr; // ns Cs per cdp
  const size_t traces_bytes = ttraces * sizeof(real);
  gx    = (real*) malloc (traces_bytes);
  gy    = (real*) malloc (traces_bytes);
  sx    = (real*) malloc (traces_bytes);
  sy    = (real*) malloc (traces_bytes);
  scalco= (real*) malloc (traces_bytes);
  cdpsmpl = (real*) malloc (sizeof(real)*ntrs*ns);

  memcpy(gx    , h_gx    , traces_bytes);
  memcpy(gy    , h_gy    , traces_bytes);
  memcpy(sx    , h_sx    , traces_bytes);
  memcpy(sy    , h_sy    , traces_bytes);
  memcpy(scalco, h_scalco, traces_bytes);

  c    = (real*) malloc (sizeof(real)*nc      );
  h    = (real*) malloc (sizeof(real)*ttraces );
  num  = (real*) malloc (sizeof(real)*ns*nc   );
  stt  = (real*) malloc (sizeof(real)*ns*nc   );
  ctr  = (int *) malloc (sizeof(int )*ncdps*ns);
  str  = (real*) malloc (sizeof(real)*ncdps*ns);
  stk  = (real*) malloc (sizeof(real)*ncdps*ns);

  //
  // HOST REGION
  //
  auto beg = std::chrono::high_resolution_clock::now();

  // Evaluate Cs - linspace
  init_c(c, inc, c0, nc);

  // Evaluate halfoffset points in x and y coordinates
  init_half(scalco, gx, gy, sx, sy, h, ttraces);

  #pragma omp parallel num_threads(ncdps)
  {
    int cdp_id = omp_get_thread_num();
    int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id-1] : 0;
    int t_idf = ntraces_by_cdp_id[cdp_id];
    int stride = t_idf - t_id0;

    memcpy(cdpsmpl, h_samples + t_id0*ns , sizeof(real)*stride*ns);

    // Compute semblances for each c for each sample
    compute_semblances(
        h, c, cdpsmpl, num, stt, t_id0, t_idf, idt, dt, tau, w, nc, ns, cdp_id*ns);

    // Get max C for max semblance for each sample on this cdp
    redux_semblances(num, stt, ctr, str, stk, nc, cdp_id, ns, 0);

    number_of_semblances += stride;

    #ifdef DEBUG
    std::cout << "Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps) << std::endl;
    #endif
  }
  // Gets time at end of computation
  auto end = std::chrono::high_resolution_clock::now();

  //
  // END HOST REGION
  //

  // Verify
  real* h_c   = (real*) malloc (sizeof(real)*nc      );
  real* h_h   = (real*) malloc (sizeof(real)*ttraces );
  real* h_num = (real*) malloc (sizeof(real)*ns*nc   );
  real* h_stt = (real*) malloc (sizeof(real)*ns*nc   );

  // reference results
   int* r_ctr = (int*) malloc (sizeof(int )*ncdps*ns);
  real* r_str = (real*) malloc (sizeof(real)*ncdps*ns);
  real* r_stk = (real*) malloc (sizeof(real)*ncdps*ns);

  h_init_c(nc, h_c, inc, c0);

  h_init_half(ttraces, h_scalco, h_gx, h_gy, h_sx, h_sy, h_h);

  for(int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
    int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id-1] : 0;
    int t_idf = ntraces_by_cdp_id[cdp_id];

    // Compute semblances for each c for each sample
    h_compute_semblances(
        h_h, h_c, h_samples+t_id0*ns, h_num, h_stt, t_id0, t_idf, idt, dt, tau, w, nc, ns);

    // Get max C for max semblance for each sample on this cdp
    h_redux_semblances(h_num, h_stt, r_ctr, r_str, r_stk, nc, cdp_id, ns, 0);
#ifdef DEBUG
    std::cout << "Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps) << std::endl;
#endif
  }

  int err_ctr = 0, err_str = 0, err_stk = 0;
  for (int i = 0; i < ncdps*ns; i++) {
   if (r_ctr[i] != ctr[i]) err_ctr++;
   if (r_str[i] - str[i] > 1e-3) err_str++;
   if (r_stk[i] - stk[i] > 1e-3) err_stk++;
  }
  float err_ctr_rate = (float)err_ctr / (ncdps * ns);
  float err_str_rate = (float)err_str / (ncdps * ns); 
  float err_stk_rate = (float)err_stk / (ncdps * ns); 
  printf("Error rate: ctr=%e str=%e stk=%e\n",
         err_ctr_rate, err_str_rate, err_stk_rate);

  // Logs stats (semblance-traces per second)
  double time = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
  double stps = (number_of_semblances / 1e9 ) * (ns * nc / time);
  std::string stats = "Giga semblances traces per second: " + std::to_string(stps);
  LOG(INFO, stats);

#ifdef SAVE
  // Delinearizes data and save it into a *.su file
  for(int i=0; i < ncdps; i++) {
    su_trace ctr_t = gather[i].traces()[0];
    su_trace str_t = gather[i].traces()[0];
    su_trace stk_t = gather[i].traces()[0];

    ctr_t.offset() = 0;
    ctr_t.sx() = ctr_t.gx() = (gather[i].traces()[0].sx() + gather[i].traces()[0].gx()) >> 1;
    ctr_t.sy() = ctr_t.gy() = (gather[i].traces()[0].sy() + gather[i].traces()[0].gy()) >> 1;

    for(int k=0; k < ns; k++) ctr_t.data()[k] =  c[ctr[i*ns+k] % nc] / FACTOR;
    str_t.data().assign(str + i*ns, str + (i+1)*ns);
    stk_t.data().assign(stk + i*ns, stk + (i+1)*ns);

    ctr_t.fputtr(c_out);
    str_t.fputtr(s_out);
    stk_t.fputtr(stack);
  }
#endif

  free(h_gx              );
  free(h_gy              );
  free(h_sx              );
  free(h_sy              );
  free(h_scalco          );
  free(h_samples         );
  free(ntraces_by_cdp_id );
  free(gx    );
  free(gy    );
  free(sx    );
  free(sy    );
  free(scalco);
  free(cdpsmpl);
  free(h     );
  free(c     );
  free(num   );
  free(stt   );
  free(ctr   );
  free(str   );
  free(stk   );
  free(h_c   );
  free(h_h   );
  free(h_num );
  free(h_stt );
  free(r_ctr );
  free(r_str );
  free(r_stk );

  return EXIT_SUCCESS;
}