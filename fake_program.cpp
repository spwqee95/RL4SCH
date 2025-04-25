#include <iostream>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <ctime>
using json = nlohmann::json;

const int NUM_BBBES = 100;
const int STATE_DIM = 13;
const int MAX_LEVEL = 10;

float random_float(float min, float max) {
    return min + static_cast<float>(rand()) / (RAND_MAX / (max - min));
}

int main() {
    srand((unsigned)time(nullptr));

    float max_step = 100.0f;

    for (int i = 0; i < NUM_BBBES; ++i) {
        // --- Local attributes ---
        int type     = rand() % 5;
        int level    = rand() % MAX_LEVEL;
        int fanin    = 1 + rand() % 5;
        int fanout   = 1 + rand() % 5;

        float ox = random_float(0.0f, 100.0f);
        float oy = random_float(0.0f, 100.0f);
        float nx = random_float(0.0f, 100.0f);
        float ny = random_float(0.0f, 100.0f);

        float delta = random_float(-3.0f, 3.0f);
        float distance = std::abs(ox - nx) + std::abs(oy - ny);

        // --- Global attributes ---
        float density = random_float(0.1f, 1.0f);
        float percent_remaining = static_cast<float>(NUM_BBBES - i) / NUM_BBBES;

        // Build state vector
        std::vector<float> state = {
            static_cast<float>(type),
            static_cast<float>(level),
            static_cast<float>(fanin),
            static_cast<float>(fanout),
            ox, oy,
            nx, ny,
            delta,
            distance,
            max_step,
            density,
            percent_remaining
        };

        json msg_out = {
            {"state", state},
            {"reward", -delta},   // shaped reward
            {"done", false}
        };

        std::cout << msg_out.dump() << std::endl;

        // Wait for action
        std::string line;
        if (!std::getline(std::cin, line)) break;
        json msg_in = json::parse(line);
        int action = msg_in.value("action", 0);

        // Simulate effect: relocation improves max_step
        if (action == 1 && delta < 0)
            max_step += delta;
    }

    // Done
    json final = {
        {"state", std::vector<float>(STATE_DIM, 0.0f)},
        {"reward", 0.0},
        {"done", true},
        {"final_max_step", max_step}
    };
    std::cout << final.dump() << std::endl;
    return 0;
}
