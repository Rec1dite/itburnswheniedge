// Dino Gironi (u21630276)
#include "edge.h"
#include "anneal.h"

int main() {
	try {
		info();

		array inp = loadImage("../Dataset/Img0.png", true) / 255.0;
		inp = rgb2gray(inp);
		array can = canny(inp, AF_CANNY_THRESHOLD_MANUAL, LOW_THRESH_RAT, HIGH_THRESH_RAT).as(f32);

		// 1) Noise reduction
		//array blur = Pipeline::noiseReduction(inp);
		// 2) Gradient calculation
		//array mag, dir;
		//std::tie(mag, dir) = Pipeline::gradientCalculation(blur);
		// 3) Edge thinning
		//array grad = Pipeline::edgeThinning(dir, mag);
		// 4) Thresholding
		//array thresh = Pipeline::thresholding(grad);
		// 5) Edge linking
		//array hough, houghTop, linked;
		//std::tie(hough, houghTop, linked) = Pipeline::edgeLinking(thresh, grad);
		// Line drawing
		//array canvas = constant(0, 50, 50);
		//canvas = Pipeline::drawLine(canvas, 10, 10, 30, 40);
		//canvas = Pipeline::drawLine(canvas, 5, 40, 8, 10);
		//canvas = Pipeline::drawLine(canvas, 20, 20, 40, 20);
		//canvas = Pipeline::drawLine(canvas, 1, 1, 1, 30);

		//array result = Pipeline::compute(inp);

		Window win(300, 300, "COS791 A1");
		Anneal opt(10000, 0.001, 0.99, 400, inp, can, win);
		Config best = opt.optimize();
		
		std::cout
			<< std::endl << std::endl
			<< "----- Best solution -----"					<< std::endl
			<< "Score: "			<< best.score			<< std::endl
			<< "LowThreshRat: "		<< best.lowThreshRat	<< std::endl
			<< "HighThreshRat: "	<< best.highThreshRat	<< std::endl
			<< "Blur: "				<< best.blur			<< std::endl
			<< "HoughThetaCount: "	<< best.houghThetaCount << std::endl
			<< "HoughRhoCount: "	<< best.houghRhoCount	<< std::endl
			<< "HoughK: "			<< best.houghK			<< std::endl;

		Stages stages = Pipeline::compute(inp, best);

		while (!win.close()) {
            Pipeline::display(
                win, inp, can, stages.blur, stages.dir,
				stages.mag, stages.grad, stages.thresh, stages.hough,
				stages.houghTop, stages.linked
            );
		}

		//=============== Display ===============//
		//array out = dX;

		//int numImgs = 10;
		//int w = std::min(4, numImgs);
		//int h = std::ceil(numImgs / 4.0f);
		//Window win(w * 300, h * 300, "COS791 A1");
		//Window win(1, 1, "COS791 A1");

		//while (!win.close()) {
		//	Pipeline::display(win, inp, can, blur, dir, mag, grad, thresh, hough, houghTop, linked);
		//}

		//int numImgs = 10;
		//int w = std::min(4, numImgs);
		//int h = std::ceil(numImgs / 4.0f);
		//while (!win.close()) {
		//	win.grid(h, w);
		//	win(0, 0).image(inp, "Input");
		//	win(1, 0).image(can, "Canny");
		//	win(0, 1).image(blur, "blur");
		//	//win(1, 1).image(dY, "dY");

		//	win(0, 2).image(dir / 4.0, "dir");
		//	win(1, 2).image(mag, "mag");
		//	win(0, 3).image(grad, "grad");
		//	win(1, 3).image(thresh.as(f32), "thresh");
		//	win(2, 0).image(transpose(hough), "hough");
		//	win(2, 1).image(transpose(houghTop), "houghTop");
		//	win(2, 2).image(linked, "linked");
		//	win(2, 3).image(canvas, "canvas");

		//	win.show();
		//}

    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

	return 0;
}