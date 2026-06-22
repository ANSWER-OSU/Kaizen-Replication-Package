class Projectile {
private:
    float m_angle;
    float m_velocity;
    float m_range;
    float m_totalTime;
    float m_maxHeight;

public:
    Projectile() {
        m_angle = 0;
        m_velocity = 0;
        m_range = 0;
        m_totalTime = 0;
        m_maxHeight = 0;
    }

    Projectile(float angle, float velocity, float range, float time, float maxheight) {
        m_angle = angle;
        m_velocity = velocity;
        m_range = range;
        m_totalTime = time;
        m_maxHeight = maxheight;
    }

    float getangle() const { return m_angle; }
    float getvelocity() const { return m_velocity; }
    void setRangeandTime(float frange, float ttime, float angle_s, float velocity_s, float height_s) {
        m_range = frange;
        m_totalTime = ttime;
        m_angle = angle_s;
        m_velocity = velocity_s;
        m_maxHeight = height_s;
    }
    float getRange() const { return m_range; }
    float gettotalTime() const { return m_totalTime; }
    float getmaxHeight() const { return m_maxHeight; }
    friend bool operator!=(const Projectile& a, const Projectile& b) {
        float err_bound = 1.f;
        return fabsf(a.m_angle - b.m_angle) > err_bound ||
            fabsf(a.m_velocity - b.m_velocity) > err_bound ||
            fabsf(a.m_range - b.m_range) > err_bound ||
            fabsf(a.m_totalTime - b.m_totalTime) > err_bound ||
            fabsf(a.m_maxHeight - b.m_maxHeight) > err_bound;
    }
    friend std::ostream& operator<<(std::ostream& out, const Projectile& obj) {
        out << "Angle: " << obj.getangle() << " Velocity: " << obj.getvelocity()
            << " Range: " << obj.getRange() << " Total time: " << obj.gettotalTime()
            << " Maximum Height: " << obj.getmaxHeight() << "\n";
        return out;
    }
};