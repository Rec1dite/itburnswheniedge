#pragma once

#include <iostream>
#define NOMINMAX
#include <arrayfire.h>
#include <tuple>

using namespace af;

// Default values
const float LOW_THRESH_RAT = 0.5;
const float HIGH_THRESH_RAT = 0.2;
const float BLUR = 0.1;
const int HOUGH_THETA_COUNT = 180 * 5;
const int HOUGH_RHO_COUNT = 50 * 5;
const int HOUGH_K = 20;

struct Config {
	float score = -1; // Memoized
	float lowThreshRat = 0.5;
	float highThreshRat = 0.2;
	float blur = 0.1;
	int houghThetaCount = 180 * 5;
	int houghRhoCount = 50 * 5;
	int houghK = 20;
};

struct Stages {
	array blur, mag, dir, grad, thresh, hough, houghTop, linked;
};

class Pipeline {
public:

	// Full pipeline
	static Stages compute(const array& inp, const Config& conf) {
		array mag, dir, hough, houghTop, linked;
		array blur = Pipeline::noiseReduction(inp, conf.blur);
		std::tie(mag, dir) = Pipeline::gradientCalculation(blur);
		array grad = Pipeline::edgeThinning(dir, mag);
		array thresh = Pipeline::thresholding(grad, conf.highThreshRat, conf.lowThreshRat);
		std::tie(hough, houghTop, linked) = Pipeline::edgeLinking(thresh, grad, conf.houghThetaCount, conf.houghRhoCount, conf.houghK);

		return { blur, mag, dir, grad, thresh, hough, houghTop, linked };
	}

	static std::pair<array, array> prewitt1(const array& in) {
		static float h1[] = { 1, 1, 1 };
		static float h2[] = { -1, 0, 1 };
		static array colf(3, 1, h1);
		static array rowf(3, 1, h2);

		// Find gradients
		array Gy = convolve(rowf, colf, in);
		array Gx = convolve(colf, rowf, in);

		return { hypot(Gx, Gy), atan2(Gy, Gx) };
	}

	//=============== Noise reduction ===============//
	static array noiseReduction(const array& in, float blurAmt = BLUR) {
		array gauss = gaussianKernel(3, 3, 1, 1); // Gaussian kernel
		array blur = convolve(in, gauss); // Gaussian blur

		blur = bilateral(in, blurAmt, 10); // Bilateral edge-preserving noise filter

		return blur;
	}

	//=============== Gradient calculation ===============//
	static std::pair<array, array> gradientCalculation(const array& in) {
		//array sbl = sobel(blur);

		array dX, dY;
		sobel(dX, dY, in);
		array mag = hypot(dY, dX);
		array dir = abs(atan2(dY, dX)); // ignore sign

		//std::tie(mag, dir) = prewitt1(blur);
		return { mag, dir };
	}

	//=============== Edge thinning ===============//
	static array edgeThinning(const array& dir, const array& mag) {
		// Round dir to nearest pi/4, remap to [0, 3] cases
		array compass = round(dir / (Pi / 4.0)) % 4; // (0 -> 0|Pi ; 1 -> Pi/4 ; 2 -> Pi/2 ; 3 -> 3/4*Pi)

		// Non-max Suppression
		array n1 = mag;
		array n2 = mag;
		n1 =
			(compass == 0) * shift(mag, 1, 0) +
			(compass == 1) * shift(mag, -1, 1) +
			(compass == 2) * shift(mag, 0, 1) +
			(compass == 3) * shift(mag, 1, 1);

		n2 =
			(compass == 0) * shift(mag, -1, 0) +
			(compass == 1) * shift(mag, 1, -1) +
			(compass == 2) * shift(mag, 0, -1) +
			(compass == 3) * shift(mag, -1, -1);

		array grad = mag;
		grad(mag < n1 || mag < n2) = 0;
		grad /= max<float>(grad); // Normalize

		return grad;
	}

	//=============== Thresholding ===============//
	static array thresholding(const array& in, float highThreshRat = HIGH_THRESH_RAT, float lowThreshRat = LOW_THRESH_RAT) {
		float highThresh = highThreshRat;
		float lowThresh = lowThreshRat * highThresh;
		//std::cout << "High thresh: " << highThresh << std::endl;
		//std::cout << "Low thresh: " << lowThresh << std::endl;

		array thresh = constant(false, in.dims(), b8); // Default to non-edge
		// Strong edges
		thresh(in >= highThresh) = true;

		// Weak edges (Apply hysteresis)
		array weak = (in < highThresh) && (in >= lowThresh);

		thresh = thresh || (weak && (
			shift(thresh, -1, 1) || shift(thresh, 0, 1) || shift(thresh, 1, 1) ||
			shift(thresh, -1, 0) || shift(thresh, 0, 0) || shift(thresh, 1, 0) ||
			shift(thresh, -1, -1) || shift(thresh, 0, -1) || shift(thresh, 1, -1)
			));

		return thresh;
	}

	//=============== Edge linking ===============//
	static std::tuple<array, array, array> edgeLinking(const array& thresh, const array& grad, int thetaCount = HOUGH_THETA_COUNT, int rhoCount = HOUGH_RHO_COUNT, int k = HOUGH_K) {
		array active = where(thresh);

		// [0, Pi) x (-Inf, Inf) -> [0, THETA_COUNT) x [0, RHO_COUNT)
		array hough = constant(0, thetaCount, rhoCount, f32);

		int* activeIndicesHost = active.as(s32).host<int>();

		array thetas = seq(0, rhoCount - 1);
		array nThetas = thetas.as(f32) * Pi / thetaCount;
		array coss = cos(nThetas);
		array sins = sin(nThetas);

		for (int i = 0; i < active.dims(0); i++) {
			// Get normalized x and y
			float x = (activeIndicesHost[i] % grad.dims(0)) / (float)grad.dims(0);
			float y = (activeIndicesHost[i] / grad.dims(0)) / (float)grad.dims(1);

			array rhos = (x * coss + y * sins) * (rhoCount / 2 - 1); // Compute rho values

			rhos += rhoCount / 2; // Center vertically
			rhos = round(rhos);

			array coords = thetas + rhos * thetaCount;

			hough(coords) += 1;
		}
		hough /= max<float>(hough);

		freeHost(activeIndicesHost);

		// Find top k values
		array top;
		array topIndices; // Indices in rho-theta space where common lines exist
		topk(top, topIndices, flat(hough), k);

		// Print
		//std::cout << "TOP K:" << std::endl;
		//af_print(top);

		//std::cout << "INDICES:" << std::endl;
		//af_print(topIndices);

		// Decompose indices back into rho, theta
		array topRhos = topIndices / thetaCount;
		array topThetas = topIndices % thetaCount;

		array houghTop = constant(0, thetaCount, rhoCount, f32);
		houghTop(topIndices) = 1;
		houghTop(topThetas, span) += 0.5;
		houghTop(span, topRhos) += 0.5;

		array linked = constant(0, thresh.dims(), f32); // Initialize the linked edges array

		int* topIndicesHost = topIndices.as(s32).host<int>();

		// Draw lines found by the Hough transform
		for (int i = 0; i < k; i++) {
			//float rho = topRhos(i) - HOUGH_RHO_COUNT / 2;
			//float theta = topThetas(i) * Pi / HOUGH_THETA_COUNT;
			float rho = (topIndicesHost[i] / thetaCount) - rhoCount / 2;
			float theta = (topIndicesHost[i] % thetaCount) * Pi / thetaCount;

			// y = mx + c
			float m = -1.0f / tan(theta);
			float c = rho / sin(theta);

			// Draw line
			//linked = drawLine(linked, 0, c, linked.dims(0), m * linked.dims(0) + c);
		}

		freeHost(topIndicesHost);

		return { hough, houghTop, linked };
	}

	//=============== Line drawing ===============//
	// Bresenham's line algo
	static array drawLine(const array& in, int x1, int y1, int x2, int y2) {
		array out = in.copy();

		int dx = abs(x2 - x1);
		int dy = abs(y2 - y1);
		int steps = std::max(dx, dy);

		array t = seq(0, steps) / steps; // Create linear spaces

		// Interpolation
		array xs = (1 - t) * x1 + t * x2;
		array ys = (1 - t) * y1 + t * y2;

		xs = round(xs);
		ys = round(ys);

		// Get linear indices for the output array
		array lineIndices = xs + ys * out.dims(1);

		// Set the line value on the output image
		out(flat(lineIndices)) = 1;

		return out;
	}

	static array drawLine(const array& in, int m, int c) {
		array out = in.copy();

		array x = seq(out.dims(0));
		array y = m * x + c;

		y = round(y);

		// Get linear indices for the output array
		array lineIndices = x + y * out.dims(1);

		// Set the line value on the output image
		out(flat(lineIndices)) = 1;

		return out;
	}

	//=============== Error functions ===============//
	static float meanSquareError(const array& a, const array& b) {
		array diff = a - b;
		return sum<float>(diff * diff) / a.elements();
	}

	//=============== Display ===============//
	static void display(
		Window& win,
		const array& inp,
		const array& can,
		const array& blur,
		const array& dir,
		const array& mag,
		const array& grad,
		const array& thresh,
		const array& hough,
		const array& houghTop,
		const array& linked
	) {
		int numImgs = 10;
		int w = std::min(4, numImgs);
		int h = std::ceil(numImgs / 4.0f);
		win.setSize(w * 300, h * 300);

		win.grid(h, w);
		win(0, 0).image(inp, "Input");
		win(1, 0).image(can, "Canny");
		win(0, 1).image(blur, "blur");
		//win(1, 1).image(dY, "dY");

		win(0, 2).image(dir / 4.0, "dir");
		win(1, 2).image(mag, "mag");
		win(0, 3).image(grad, "grad");
		win(1, 3).image(thresh.as(f32), "thresh");
		win(2, 0).image(transpose(hough), "hough");
		win(2, 1).image(transpose(houghTop), "houghTop");
		win(2, 2).image(linked, "linked");

		win.show();
	}
};
