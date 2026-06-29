// ...

void hotspot3d(
    const float*__restrict__ tIn,
    const float*__restrict__ pIn,
          float*__restrict__ tOut,
    const int numCols,
    const int numRows,
    const int layers,
    const float ce,
    const float cw,
    const float cn,
    const float cs,
    const float ct,
    const float cb,
    const float cc,
    const float stepDivCap)
{
  // ...

  int W = (i == 0)         ? c : c - 1;
  int E = (i == numCols-1)     ? c : c + 1;
  int N = (j == 0)         ? c : c - numCols;
  int S = (j == numRows-1)     ? c : c + numCols;

  float temp1, temp2, temp3;
  temp1 = temp2 = tIn[c];
  temp3 = tIn[c+xy];
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
    + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
  c += xy;
  W += xy;
  E += xy;
  N += xy;
  S += xy;

  for (int k = 1; k < layers-1; ++k) {
    temp1 = temp2;
    temp2 = temp3;
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
      + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;
  }
  temp1 = temp2;
  temp2 = temp3;
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
    + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;

  // ...
}

int main(int argc, char** argv)
{
  // ...

  float* tIn   = (float*) calloc(size,sizeof(float));
  float* pIn   = (float*) calloc(size,sizeof(float));
  float* tCopy = (float*) malloc(size * sizeof(float));
  float* tOut  = (float*) calloc(size,sizeof(float));

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  memcpy(tCopy,tIn, size * sizeof(float));

  long long start = get_time();

  for(int j = 0; j < iterations; j++)
  {
    hotspot3d(
        tIn, pIn, tOut, numCols, numRows, layers,
        ce, cw, cn, cs, ct, cb, cc, stepDivCap);

    float* temp = tIn;
    tIn = tOut;
    tOut = temp;
  }

  long long stop = get_time();

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tOut,answer,numRows*numCols*layers);
  float time = (float)((stop - start)/(1000.0 * 1000.0));
  printf("Device offloading time: %.3f (s)\n",time);
  printf("Root-mean-square error: %e\n",acc);

  writeoutput(tOut,numRows,numCols,layers,ofile);

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  return 0;
}

// ...