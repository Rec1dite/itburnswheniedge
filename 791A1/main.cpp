// Dino Gironi (u21630276)

#include <iostream>
#define NOMINMAX
#include <arrayfire.h>

using namespace af;

const float LOW_THRESH_RAT = 0.5;
const float HIGH_THRESH_RAT = 0.2;
const float BLUR = 0.1;
const int HOUGH_THETA_COUNT = 180 * 5;
const int HOUGH_RHO_COUNT = 50 * 5;
const int HOUGH_K = 20;

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

		thresh = thresh || (weak && (
			shift(thresh, -1, 1) || shift(thresh, 0, 1) || shift(thresh, 1, 1) ||
			shift(thresh, -1, 0) || shift(thresh, 0, 0) || shift(thresh, 1, 0) ||
			shift(thresh, -1, -1) || shift(thresh, 0, -1) || shift(thresh, 1, -1)
			));

		//=============== Edge linking ===============//
		array active = where(thresh);

		// [0, Pi) x (-Inf, Inf) -> [0, THETA_COUNT) x [0, RHO_COUNT)
		array hough = constant(0, HOUGH_THETA_COUNT, HOUGH_RHO_COUNT, f32);

		int* indices = active.as(s32).host<int>();

		array thetas = seq(0, HOUGH_THETA_COUNT - 1);
		array nThetas = thetas.as(f32) * Pi / HOUGH_THETA_COUNT;
		array coss = cos(nThetas);
		array sins = sin(nThetas);

		for (int i = 0; i < active.dims(0); i++) {
			// Get normalized x and y
			float x = (indices[i] % grad.dims(0)) / (float)grad.dims(0);
			float y = (indices[i] / grad.dims(0)) / (float)grad.dims(1);

			array rhos = (x * coss + y * sins) * (HOUGH_RHO_COUNT / 2 - 1); // Compute rho values

			rhos += HOUGH_RHO_COUNT / 2; // Center vertically
			rhos = round(rhos);

			array coords = thetas + rhos * HOUGH_THETA_COUNT;

			hough(coords) += 1;
		}
		hough /= max<float>(hough);

		freeHost(indices);

		// Find top k values
		array top;
		array topIndices;
		topk(top, topIndices, flat(hough), HOUGH_K);

		// Print
		//std::cout << "TOP K:" << std::endl;
		//af_print(top);

		//std::cout << "INDICES:" << std::endl;
		//af_print(topIndices);

		// Decompose indices back into rho, theta
		array topRhos = topIndices / HOUGH_THETA_COUNT;
		array topThetas = topIndices % HOUGH_THETA_COUNT;

		array houghTop = constant(0, HOUGH_THETA_COUNT, HOUGH_RHO_COUNT, f32);
		//houghTop(topIndices) = 1;
		houghTop(topThetas, span) += 0.5;
		houghTop(span, topRhos) += 0.5;

		array linked = constant(0, thresh.dims(), f32); // Initialize the linked edges array

		// Draw lines found by the Hough transform
		for (int i = 0; i < topIndices.elements(); ++i) {
			float r = topRhos(i).scalar<float>();
			float t = topThetas(i).scalar<float>();
			float cos_t = cos(t);
			float sin_t = sin(t);

			// Iterate through image dimensions and mark the pixels forming the lines
			for (int x = 0; x < thresh.dims(1); ++x) {
				for (int y = 0; y < thresh.dims(0); ++y) {
					if (std::abs(x * cos_t + y * sin_t - r) < 1.0f) {
						linked(y, x) = 1.0f;
					}
				}
			}
		}

		//=============== Display ===============//
		array out = dX;

		int numImgs = 10;
		int w = std::min(4, numImgs);
		int h = std::ceil(numImgs / 4.0f);
		Window win(w * 300, h * 300, "COS791 A1");
		while (!win.close()) {
			win.grid(h, w);
			win(0, 0).image(inp, "Input");
			win(1, 0).image(can, "Canny");
			win(0, 1).image(blur, "blur");
			win(1, 1).image(dY, "dY");
			win(0, 2).image(dir / 4.0, "dir");
			win(1, 2).image(mag, "mag");
			win(0, 3).image(grad, "grad");
			win(1, 3).image(thresh.as(f32), "thresh");
			win(2, 0).image(linked, "linked");
			win(2, 1).image(transpose(hough), "hough");
			win(2, 2).image(transpose(houghTop), "houghTop");
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