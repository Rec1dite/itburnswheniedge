// Dino Gironi (u21630276)

#include <iostream>
#define NOMINMAX
#include <arrayfire.h>

using namespace af;

const float LOW_THRESH_RAT = 0.5;
const float HIGH_THRESH_RAT = 0.4;
const float BLUR = 0.1;

 void prewitt1(array &mag, array &dir, const array &in) {
     static float h1[] = {1, 1, 1};
     static float h2[] = {-1, 0, 1};
     static array colf(3, 1, h1);
     static array rowf(3, 1, h2);
  
     // Find gradients
     array Gy = convolve(rowf, colf, in);
     array Gx = convolve(colf, rowf, in);
  
     mag = hypot(Gx, Gy);
     dir = atan2(Gy, Gx);
 }

int main() {
    try {
		info();

		array inp = loadImage("../Dataset/jake.jpg", true) / 255.0;
		inp = rgb2gray(inp);
		array can = canny(inp, AF_CANNY_THRESHOLD_MANUAL, LOW_THRESH_RAT, HIGH_THRESH_RAT).as(f32);

		//=============== Noise reduction ===============//
		array gauss = gaussianKernel(3, 3, 1, 1); // Gaussian kernel
		array blur = convolve(inp, gauss); // Gaussian blur
		blur = bilateral(inp, BLUR, 10); // Bilateral edge-preserving noise filter

		//=============== Gradient calculation ===============//
		//array sbl = sobel(blur);

		array dX, dY;
		sobel(dX, dY, blur);
		array mag = hypot(dY, dX);
		array dir = abs(atan2(dY, dX)); // ignore sign

		//prewitt1(mag, dir, blur);

		//=============== Edge thinning ===============//
		//array grad = abs(dX + dY); // approximative

		// Round dir to nearest pi/4, remap to [0, 3] cases
		dir = round(dir / (Pi / 4.0)) % 4; // (0 -> 0|Pi ; 1 -> Pi/4 ; 2 -> Pi/2 ; 3 -> 3/4*Pi)
		//dir = round(dir / (4.0 / Pi)) % 4; // (0 -> 0|Pi ; 1 -> Pi/4 ; 2 -> Pi/2 ; 3 -> 3/4*Pi)

		// Non-max Suppression
		array n1 = mag;
		array n2 = mag;
		n1 =
			(dir == 0) * shift(mag, 1, 0) +
			(dir == 1) * shift(mag, -1, 1) +
			(dir == 2) * shift(mag, 0, 1) +
			(dir == 3) * shift(mag, 1, 1);

		n2 =
			(dir == 0) * shift(mag, -1, 0) +
			(dir == 1) * shift(mag, 1, -1) +
			(dir == 2) * shift(mag, 0, -1) +
			(dir == 3) * shift(mag, -1, -1);

		array grad = mag;
		grad(mag < n1 || mag < n2) = 0;
		grad /= max<float>(grad); // Normalize

		//=============== Thresholding ===============//

		float highThresh = HIGH_THRESH_RAT;
		float lowThresh = LOW_THRESH_RAT * highThresh;
		std::cout << "High thresh: " << highThresh << std::endl;
		std::cout << "Low thresh: " << lowThresh << std::endl;

		array thresh = constant(false, grad.dims(), b8); // Default to non-edge
		// Strong edges
		thresh(grad >= highThresh) = true;

		// Weak edges (Apply hysteresis)
		array weak = (grad < highThresh) && (grad >= lowThresh);

		thresh = weak && (
			shift(thresh, -1, 1)	|| shift(thresh, 0, 1)	|| shift(thresh, 1, 1) ||
			shift(thresh, -1, 0)	|| shift(thresh, 0, 0)	|| shift(thresh, 1, 0) ||
			shift(thresh, -1, -1)	|| shift(thresh, 0, -1)	|| shift(thresh, 1, -1)
		);
		thresh = thresh.as(f32);

		//=============== Edge linking ===============//


		//=============== Display ===============//
		array out = dX;

		int numImgs = 8;
		int w = std::min(4, numImgs);
		int h = std::ceil(numImgs / 4.0f);
		Window win(w*300, h*300, "COS791 A1");
		while (!win.close()) {
			win.grid(h, w);
			win(0, 0).image(inp, "Input");
			win(1, 0).image(can, "Canny");
			win(0, 1).image(blur, "blur");
			win(1, 1).image(dY, "dY");
			win(0, 2).image(dir / 4.0, "dir");
			win(1, 2).image(mag, "mag");
			win(0, 3).image(grad, "grad");
			win(1, 3).image(thresh, "thresh");
			//win(0, 3).image((dir == 0).as(f32), "dir0");
			//win(1, 3).image((dir == 1).as(f32), "dir1");
			//win(2, 3).image((dir == 2).as(f32), "dir2");
			//win(3, 3).image((dir == 3).as(f32), "dir3");
			win.show();
		}

    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

	return 0;
}