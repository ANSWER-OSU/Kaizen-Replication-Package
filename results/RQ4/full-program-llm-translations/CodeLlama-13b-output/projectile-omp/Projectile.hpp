using namespace std;

class Projectile {
private:
    float m_angle;
    float m_velocity;
    float m_range;
    float m_totalTime;
    float m_maxHeight;

public:
    Projectile() {
        m_angle = 0.0f;
        m_velocity = 0.0f;
        m_range = 0.0f;
        m_totalTime = 0.0f;
        m_maxHeight = 0.0f;
    }

    Projectile(float angle, float velocity, float range, float time, float maxheight) {
        m_angle = angle;
        m_velocity = velocity;
        m_range = range;
        m_totalTime = time;
        m_maxHeight = maxheight;
    }

    float getangle() { return m_angle; }
    float getvelocity() { return m_velocity; }

    void setRangeandTime(float frange, float ttime, float angle_s, float velocity_s, float height_s) {
        m_range = frange;
        m_totalTime = ttime;
        m_angle = angle_s;
        m_velocity = velocity_s;
        m_maxHeight = height_s;
    }

    float getRange() { return m_range; }
    float gettotalTime() { return m_totalTime; }
    float getmaxHeight() { return m_maxHeight; }

    friend bool operator!=(const Projectile& a, const Projectile& b) {
        float err_bound = 1.f;
        return abs(a.m_angle - b.m_angle) > err_bound || abs(a.m_velocity - b.m_velocity) > err_bound || abs(a.m_range - b.m_range) > err_bound || abs(a.m_totalTime - b.m_totalTime) > err_bound || abs(a.m_maxHeight - b.m_maxHeight) > err_bound;
    }

    friend ostream& operator<<(ostream& out, const Projectile& obj) {
        out << "Angle: " << obj.getangle() << " Velocity: " << obj.getvelocity() << " Range: " << obj.getRange() << " Total time: " << obj.gettotalTime() << " Maximum Height: " << obj.getmaxHeight() << "\n";
        return out;
    }
};