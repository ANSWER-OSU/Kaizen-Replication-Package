void chemv(float alpha_re, float alpha_im, float beta_re, float beta_im,
               struct ComplexFloat AT[AT_SIZE], struct ComplexFloat X[X_SIZE],
               struct ComplexFloat Y[Y_SIZE]) {
  #pragma omp target teams distribute parallel for
  for (int i0 = 0; i0 <= (N - 1); i0 += 1) {
    float var5_Re;
    float var5_Im;
    var5_Re =
      ((Y[i0 * INCY + 0].Re * beta_re) - (Y[i0 * INCY + 0].Im * beta_im));
    var5_Im =
      ((Y[i0 * INCY + 0].Im * beta_re) + (Y[i0 * INCY + 0].Re * beta_im));
    Y[i0 * INCY + 0].Re = var5_Re;
    Y[i0 * INCY + 0].Im = var5_Im;
  }

  #pragma omp target teams distribute parallel for collapse(2)
  for (int i1 = 0; i1 <= ((N - 1) + 1) - 1; i1 += 1) {
    float var2_Re;
    float var3_Im;
    float var2_Im;
    float var4_Im;
    float var4_Re;
    float var3_Re;
    var2_Re = (alpha_re * AT[i1 * LDAT + i1].Re);
    var2_Im = (alpha_im * AT[i1 * LDAT + i1].Re);
    var3_Re = ((var2_Re * X[i1 * INCX + 0].Re) - (var2_Im * X[i1 * INCX + 0].Im));
    var3_Im = ((var2_Im * X[i1 * INCX + 0].Re) + (var2_Re * X[i1 * INCX + 0].Im));
    var4_Re = (Y[i1 * INCY + 0].Re + var3_Re);
    var4_Im = (Y[i1 * INCY + 0].Im + var3_Im);
    Y[i1 * INCY + 0].Re = var4_Re;
    Y[i1 * INCY + 0].Im = var4_Im;
  }

  #pragma omp target teams distribute parallel for collapse(2)
  for (int i2 = 0; i2 <= ((N - 1) - 1); i2 += 1) {
    for (int i3 = 0; i3 <= (N - 1) - (1 + i2); i3 += 1) {
      float var99_Re;
      float var96_Re;
      float var98_Im;
      float var96_Im;
      float var94_Im;
      float var95_Im;
      float var94_Re;
      float var95_Re;
      float var97_Im;
      float var99_Im;
      float var97_Re;
      float var98_Re;
      var94_Re = ((alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Re) -
          (alpha_im * (-AT[i2 * LDAT + ((1 + i2) + i3)].Im)));
      var94_Im = ((alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Re) +
          (alpha_re * (-AT[i2 * LDAT + ((1 + i2) + i3)].Im)));
      var95_Re = ((var94_Re * X[((i3 + i2) + 1) * INCX + 0].Re) -
          (var94_Im * X[((i3 + i2) + 1) * INCX + 0].Im));
      var95_Im = ((var94_Im * X[((i3 + i2) + 1) * INCX + 0].Re) +
          (var94_Re * X[((i3 + i2) + 1) * INCX + 0].Im));
      var96_Re = (Y[i2 * INCY + 0].Re + var95_Re);
      var96_Im = (Y[i2 * INCY + 0].Im + var95_Im);
      Y[i2 * INCY + 0].Re = var96_Re;
      Y[i2 * INCY + 0].Im = var96_Im;
      var97_Re = ((alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Re) -
          (alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Im));
      var97_Im = ((alpha_im * AT[i2 * LDAT + ((1 + i2) + i3)].Re) +
          (alpha_re * AT[i2 * LDAT + ((1 + i2) + i3)].Im));
      var98_Re = ((var97_Re * X[i2 * INCX + 0].Re) -
          (var97_Im * X[i2 * INCX + 0].Im));
      var98_Im = ((var97_Im * X[i2 * INCX + 0].Re) +
          (var97_Re * X[i2 * INCX + 0].Im));
      var99_Re = (Y[((i3 + i2) + 1) * INCY + 0].Re + var98_Re);
      var99_Im = (Y[((i3 + i2) + 1) * INCY + 0].Im + var98_Im);
      Y[((i3 + i2) + 1) * INCY + 0].Re = var99_Re;
      Y[((i3 + i2) + 1) * INCY + 0].Im = var99_Im;
    }
  }
}