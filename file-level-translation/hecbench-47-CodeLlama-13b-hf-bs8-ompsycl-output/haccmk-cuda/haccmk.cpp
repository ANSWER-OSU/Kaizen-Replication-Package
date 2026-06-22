void haccmk (
    const int repeat,
    const int n1,
    const int n2,
    const float* xx, 
    const float* yy,
    const float* zz,
    const float* mass,
    float* vx2,
    float* vy2,
    float* vz2,
    const float fsrmax,
    const float mp_rsm,
    const float fcoeff ) 
{
  #pragma omp target teams distribute parallel for num_teams(n1) num_threads(256)
  for (int i = 0; i < n1; i++) {
    const float ma0 = 0.269327f; 
    const float ma1 = -0.0750978f; 
    const float ma2 = 0.0114808f; 
    const float ma3 = -0.00109313f; 
    const float ma4 = 0.0000605491f; 
    const float ma5 = -0.00000147177f;

    float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

    xi = 0.f; 
    yi = 0.f;
    zi = 0.f;

    float xxi = xx[i];
    float yyi = yy[i];
    float zzi = zz[i];

    for ( int j = 0; j < n2; j++ ) {
      dxc = xx[j] - xxi;
      dyc = yy[j] - yyi;
      dzc = zz[j] - zzi;

      r2 = dxc * dxc + dyc * dyc + dzc * dzc;

      //if ( r2 < fsrmax ) m = mass[j]; else m = 0.f;
      m = mass[j] * (r2 < fsrmax);

      f = r2 + mp_rsm;
      f = m * (1.f / (f * sqrtf(f)) - (ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))));

      xi = xi + f * dxc;
      yi = yi + f * dyc;
      zi = zi + f * dzc;
    }

    vx2[i] += xi * fcoeff;
    vy2[i] += yi * fcoeff;
    vz2[i] += zi * fcoeff;
  }
}