//AMP: #include <cstdio>

// AMP: Function to load an image file
unsigned char* shrLoadImage(const char *filename)
{
  return shrLoadPngFile(filename);
}

// AMP: Function to load a PNG image file
unsigned char* shrLoadPngFile(const char *filename)
{
  return shrLoadRawFile(filename, 4);
}

// AMP: Function to load a PPM image file
unsigned char* shrLoadPPMFile(const char *filename)
{
  return shrLoadRawFile(filename, 3);
}

// AMP: Function to load a PGM image file
unsigned char* shrLoadPGMFile(const char *filename)
{
  return shrLoadRawFile(filename, 2);
}

// AMP: Function to load a PPM image file
unsigned char* shrLoadPPMFile(const char *filename)
{
  return shrLoadRawFile(filename, 1);
}

// AMP: Function to load a raw data file
unsigned char* shrLoadRawFile(const char *filename, size_t size)
{
  unsigned char* data = NULL;
  size_t width, height;

  if (filename == NULL) {
    shrLog("/n> shrLoadRawFile: filename is NULL\n  Aborting Loading !!!/n");
    return NULL;
  }
  shrLog("/n> shrLoadRawFile:/n    (a)filename: <%s>, (b)size: %d/n", filename, size);

  if (shrLoadPPM4ub(filename, &data, &width, &height) != shrTRUE)
  {
    shrLog("/n    Unable to load image file: %s\n    Aborting Loading !!!/n", filename);
    return NULL;
  }

  return data;
}

//AMP: Function to load a PPM image file
shrBOOL shrLoadPPM4