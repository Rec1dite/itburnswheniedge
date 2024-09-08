#pragma once

#include "edge.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>

class Anneal {
private:
    double initT;
    double finalT;
    double coolRate;
    int maxIters;
    bool debug;

    const array& source;
    const array& target;
    Window& win;


    Config randomSolution() {
        return {
            -1,
            randomFloat(0.0f, 1.0f),
			randomFloat(0.0f, 1.0f),
			randomFloat(0.001f, 10.0f),
			randomInt(50, 500),
			randomInt(10, 100),
			randomInt(2, 20)
        };
    }

    Config randomNeighbor(const Config& sol) {
        Config neighbor = sol;
        neighbor.score = -1;
        neighbor.lowThreshRat       += randomFloat(-0.1f, 0.1f);
        neighbor.highThreshRat      += randomFloat(-0.1f, 0.1f);
        neighbor.blur               += randomFloat(-1.0f, 1.0f);
        neighbor.houghThetaCount    += randomInt(-50, 50);
		neighbor.houghRhoCount      += randomInt(-10, 10);
        neighbor.houghK             += randomInt(-10, 10);

        // Clamp values
		neighbor.lowThreshRat = clamp(neighbor.lowThreshRat, 0.0f, 1.0f);
        neighbor.highThreshRat = clamp(neighbor.highThreshRat, 0.0f, 1.0f);
		neighbor.blur = clamp(neighbor.blur, 0.001f, 10.0f);
		neighbor.houghThetaCount = clamp(neighbor.houghThetaCount, 50, 2000);
		neighbor.houghRhoCount = clamp(neighbor.houghRhoCount, 10, 1000);
		neighbor.houghK = clamp(neighbor.houghK, 2, 30);

        return neighbor;
    }

	template <typename T>
	T clamp(T val, T min, T max) {
		return std::max(min, std::min(val, max));
	}

    float randomFloat(float min, float max) {
        return min + (float)(std::rand()) / ((float)RAND_MAX / (max - min));
    }

	int randomInt(int min, int max) {
		return min + (std::rand() % (max - min + 1));
	}

    // Objective function
    double score(Config& sol, bool show = true) {
        if (sol.score != -1) { return sol.score; }

        Stages stages = Pipeline::compute(source, sol);
        array result = stages.thresh;
        float score = Pipeline::meanSquareError(source, target);
		sol.score = score;

        if (debug && show) {
            Pipeline::display(
                win, source, target, stages.blur, stages.dir,
				stages.mag, stages.grad, stages.thresh, stages.hough,
				stages.houghTop, stages.linked
            );
        }

        return score;
    }

public:

    Anneal(double initT, double finalT, double coolRate, int maxIters, const array& source, const array& target, Window& win, bool debug = true)
        : initT(initT), finalT(finalT), coolRate(coolRate), maxIters(maxIters), source(source), target(target), win(win), debug(debug) {
        std::srand(std::time(nullptr));
    }

    Config optimize() {
        //----- Create random initial solution -----//
        Config current = randomSolution();
        Config best = current;
        double currT = initT;

        for (int i = 0; i < maxIters && currT > finalT; i++) {
            std::cout << std::endl;
            std::cout << "[" << i << "] T: " << currT << std::endl;

            //----- Find random neighbour -----//
            Config neighbor = randomNeighbor(current);

            //----- Compute objective delta -----//
            double delta = score(neighbor) - score(current);

            //----- Accept/reject solution -----//
            if (delta < 0 || (std::exp(-delta / currT) > randomFloat(0.0, 1.0))) {
				std::cout << "ACCEPTED: " << delta << std::endl;
                current = neighbor;
            }

            //----- Update best solution -----//
            if (score(current) < score(best)) {
				std::cout << "NEW BEST FOUND:"      << std::endl;
				std::cout << "Score: "              << current.score << std::endl;
				std::cout << "LowThreshRat: "       << current.lowThreshRat << std::endl;
				std::cout << "HighThreshRat: "      << current.highThreshRat << std::endl;
				std::cout << "Blur: "               << current.blur << std::endl;
				std::cout << "HoughThetaCount: "    << current.houghThetaCount << std::endl;
				std::cout << "HoughRhoCount: "      << current.houghRhoCount << std::endl;
				std::cout << "HoughK: "             << current.houghK << std::endl;
                best = current;
            }

            //----- Cool down -----//
            currT *= coolRate;
        }

        return best;
    }
};