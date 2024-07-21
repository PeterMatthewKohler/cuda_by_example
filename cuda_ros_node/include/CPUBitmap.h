#ifndef __CPUBITMAP_H__
#define __CPUBITMAP_H__

#define DIM 1024

struct CPUBitmap {
    unsigned char    *pixels;
    int     x, y;
    void    *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap( int width, int height, void *d = NULL ) {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }

    ~CPUBitmap() {
        delete [] pixels;
    }

    // Copy constructor
    CPUBitmap( const CPUBitmap &b ) {
        x = b.x;
        y = b.y;
        pixels = new unsigned char[x * y * 4];
        // Assign values of pixels from b to this object
        for (int i = 0; i < x * y * 4; i++) {
            pixels[i] = b.pixels[i];
        }
        dataBlock = b.dataBlock;
    }

    unsigned char* get_ptr( void ) const   { return pixels; }
    long image_size( void ) const { return x * y * 4; }
};

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

#endif  // __CPUBITMAP_H__