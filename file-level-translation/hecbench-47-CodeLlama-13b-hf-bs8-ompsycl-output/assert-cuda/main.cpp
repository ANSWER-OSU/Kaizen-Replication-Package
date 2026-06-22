void testKernel(int N)
{
  #pragma omp target teams distribute parallel for
  for (int gid = 0; gid < N; gid++) {
    assert(gid < N);
  }
}

void perfKernel()
{
  #pragma omp target teams distribute parallel for
  for (int gid = 0; gid < blockDim.x * gridDim.x; gid++) {
    assert(gid <= blockDim.x * gridDim.x);
    int s = 0;
    for (int n = 1; n <= gid; n++) {
      s++; assert(s <= gid);
    }
  }
}

void perfKernel2()
{
  #pragma omp target teams distribute parallel for
  for (int gid = 0; gid < blockDim.x * gridDim.x; gid++) {
    int s = 0;
    for (int n = 1; n <= gid; n++) {
      s++; assert(s <= gid);
    }
  }
}