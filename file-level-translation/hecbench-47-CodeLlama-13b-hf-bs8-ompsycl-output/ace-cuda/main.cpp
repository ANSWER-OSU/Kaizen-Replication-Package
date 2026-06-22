__global__
void calculateForce(double phi[][DATAYSIZE][DATAZSIZE],
                    double Fx[][DATAYSIZE][DATAZSIZE],
                    double Fy[][DATAYSIZE][DATAZSIZE],
                    double Fz[][DATAYSIZE][DATAZSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0)
{

  unsigned iz = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned ix = blockIdx.z*blockDim.z + threadIdx.z;

  if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) &&
      (iz < (DATAZSIZE-1)) && (ix > (0)) &&
      (iy > (0)) && (iz > (0))) {

    double phix = GradientX(phi,dx,dy,dz,ix,iy,iz);
    double phiy = GradientY(phi,dx,dy,dz,ix,iy,iz);
    double phiz = GradientZ(phi,dx,dy,dz,ix,iy,iz);
    double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
    double c = 16.0 * W0 * epsilon;
    double w = Wn(phix,phiy,phiz,epsilon,W0);
    double w2 = SQ(w);

    Fx[ix][iy][iz] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
    Fy[ix][iy][iz] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
    Fz[ix][iy][iz] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
  }
  else
  {
    Fx[ix][iy][iz] = 0.0;
    Fy[ix][iy][iz] = 0.0;
    Fz[ix][iy][iz] = 0.0;
  }

}