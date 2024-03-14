class AbstractCellSolver 
{
public:
    virtual ~AbstractCellSolver() = default;

    virtual void Init(mfem::GridFunction& Vgf, mfem::GridFunction& sgf) const = 0;

    virtual double EvalReaction(const double V, const mfem::Vector &cells_s, int int_point, float t) const = 0;
    virtual void InternalRHS(mfem::Vector&rhs, double V, mfem::Vector &cells_s) const = 0;

    virtual void FullExplicitStep(mfem::Vector& cells_V, mfem::Vector& cells_s, float t, float Δt) const  = 0;
    virtual void InternalEulerUpdate(const double V, mfem::Vector &cells_s, int int_point, float t, double Δt) const = 0;

    virtual int InternalDim() const = 0;
    virtual int HGateIndex() const = 0;
    
    virtual void RescaleVoltage(mfem::GridFunction& φₘgf) const = 0;
};

// Implementation of https://www.frontiersin.org/articles/10.3389/fphys.2019.00721/full
class PathmanathanCordeiroGrayCellSolver final 
: public AbstractCellSolver
{
private:
    // Parameters
    float C_m = 1.0f; // [µF/cm^-2]
    //float C_m = 0.01f; // [µF/mm^-2]
    // ------ I_Na -------
    constexpr static float g_Na = 12.0f;    // [mS/µF]
    constexpr static float E_m  = -52.244f; // [mV]
    constexpr static float k_m  = 6.5472f;  // [mV]
    constexpr static float τ_m  = 0.12f;    // [ms]
    constexpr static float E_h  = -78.7f;   // [mV]
    constexpr static float k_h  = 5.93f;    // [mV]
    constexpr static float δ_h  = 0.799163; // dimensionless
    constexpr static float τ_h0 = 6.80738;  // [ms]
    // ------ I_K1 -------
    constexpr static float g_K1 = 0.73893f;  // [mS/µF]
    constexpr static float E_z  = -91.9655f; // [mV]
    constexpr static float k_z  = 12.4997f;  // [mV]
    // ------ I_to -------
    float g_to = 0.1688f*1.9;   // [mS/µF]
    constexpr static float E_r  = 14.3116f;  // [mV]
    constexpr static float k_r  = 11.462f;   // [mV]
    constexpr static float E_s  = -47.9286f; // [mV]
    constexpr static float k_s  = 4.9314f;   // [mV]
    constexpr static float τ_s  = 9.90669f;  // [ms]
    // ------ I_CaL -------
    constexpr static float g_CaL = 0.11503f; // [mS/µF]
    constexpr static float E_d   = 0.7f;     // [mV]
    constexpr static float k_d   = 4.3f;     // [mV]
    constexpr static float E_f   = -15.7f;   // [mV]
    constexpr static float k_f   = 4.6f;     // [mV]
    constexpr static float τ_f   = 30.0f;    // [ms]
    // ------ I_Kr -------
    constexpr static float g_Kr = 0.056f; // [mS/µF]
    constexpr static float E_xr = -26.6f; // [mV]
    constexpr static float k_xr = 6.5f;   // [mV]
    constexpr static float τ_xr = 334.0f; // [ms]
    constexpr static float E_y  = -49.6f; // [mV]
    constexpr static float k_y  = 23.5f;  // [mV]
    // ------- I_Ks --------
    constexpr static float g_Ks = 0.008f; // [mS/µF]
    constexpr static float E_xs = 24.6f;  // [mV]
    constexpr static float k_xs = 12.1f;  // [mV]
    constexpr static float τ_xs = 628.0f; // [ms]
    // ------- Other --------
    constexpr static float E_Na = 65.0f;  // [mV]
    constexpr static float E_K  = -85.0f; // [mV]
    constexpr static float E_Ca = 50.0f;  // [mV]

    // Helper
    static inline float sigmoid(const float v, const float E_Y, const float k_Y,
                                const float sign)
    {
        return 1.0f / (1.0f + exp(sign * (v - E_Y) / k_Y));
    }

    inline float rhs_h(const float v, const float H) const
    {
        const float τ = (2.0f * τ_h0 * exp(δ_h * (v - E_h) / k_h)) /
                        (1.0f + exp((v - E_h) / k_h));
        const float h_inf = sigmoid(v, E_h, k_h, 1.0f);

        const float b = h_inf / τ;
        const float a = -1.0f / τ;

        return a*H + b;
    }

    // derivative of (1/(1+exp((x-E)/k))-h)/((2*t*exp(delta*(x-E)/k))/(1+exp((x-E)/k))) in x
    inline float rhs_hdV(const float v, const float H) const
    {
        double term1 = -δ_h/(2.0f*k_h*τ_h0) * (1.0f - H*(exp((v-E_h)/k_h)+1.0f)) * exp(δ_h * (v - E_h) / k_h);
        double term2 = 1.0f/(2.0f*k_h*τ_h0) * (1.0f/(exp((v-E_h)/k_h)+1.0f) - H) * (exp((v-E_h-(δ_h*(v-E_h)))/k_h));
        double term3 = exp((v-E_h)/k_h-(δ_h*(v-E_h))/k_h)/(2.0f*k_h*τ_h0*exp((v-E_h)/k_h));
        return term1 + term2 + term3;
    }

    inline float rhs_hdh(const float v, const float H) const
    {
        const float τ = (2.0f * τ_h0 * exp(δ_h * (v - E_h) / k_h)) /
                        (1.0f + exp((v - E_h) / k_h));
        // const float h_inf = sigmoid(v, E_h, k_h, 1.0f);

        // const float b = h_inf / τ;
        const float a = -1.0f / τ;

        return a;
    }

    inline float update_h(const float v, const float H, const float Δt) const
    {
        const float τ = (2.0f * τ_h0 * exp(δ_h * (v - E_h) / k_h)) /
                        (1.0f + exp((v - E_h) / k_h));
        const float h_inf = sigmoid(v, E_h, k_h, 1.0f);

        const float b = h_inf / τ;
        const float a = -1.0f / τ;

        return std_clamp<float>(exp(a * Δt) * (H + b / a) - b / a, 0.0f,
                          1.0f);
    }

    inline float rhs_m(const float v, const float M) const
    {
        float τ     = τ_m;
        float m_inf = sigmoid(v, E_m, k_m, -1.0f);

        float b = m_inf / τ;
        float a = -1.0f / τ;

        return a*M + b;
    }

    inline float rhs_mdv(const float v, const float M) const
    {
        return exp(-(v-E_m)/k_m)/(k_m*τ_m*(exp(-(v-E_m)/k_m)+1)*(exp(-(v-E_m)/k_m)+1));
    }

    inline float rhs_mdm(const float v, const float M) const
    {
        float τ     = τ_m;
        // float m_inf = sigmoid(v, E_m, k_m, -1.0f);

        // float b = m_inf / τ;
        float a = -1.0f / τ;

        return a;
    }

    inline float update_m(const float v, const float M, const float Δt) const
    {
        float τ     = τ_m;
        float m_inf = sigmoid(v, E_m, k_m, -1.0f);

        float b = m_inf / τ;
        float a = -1.0f / τ;

        return std_clamp<float>(exp(a * Δt) * (M + b / a) - b / a, 0.0f,
                          1.0f);
    }

    inline float rhs_s(const float v, const float S) const
    {
        const float τ     = τ_s;
        const float s_inf = sigmoid(v, E_s, k_s, 1.0f);

        float b = s_inf / τ;
        float a = -1.0f / τ;

        return a*S + b;
    }

    inline float rhs_sdv(const float v, const float S) const
    {
        return exp((v-E_s)/k_s)/(k_s*τ_s*(exp((v-E_s)/k_s)+1)*(exp((v-E_s)/k_s)+1));
    }

    inline float rhs_sds(const float v, const float S) const
    {
        const float τ     = τ_s;
        // const float s_inf = sigmoid(v, E_s, k_s, 1.0f);

        // float b = s_inf / τ;
        float a = -1.0f / τ;

        return a;
    }

    inline float update_s(const float v, const float S, const float Δt) const
    {
        const float τ     = τ_s;
        const float s_inf = sigmoid(v, E_s, k_s, 1.0f);

        float b = s_inf / τ;
        float a = -1.0f / τ;

        return std_clamp<float>(exp(a * Δt) * (S + b / a) - b / a, 0.0f,
                          1.0f);
    }

    inline float rhs_f(const float v, const float F) const
    {
        const float τ     = τ_f;
        const float f_inf = sigmoid(v, E_f, k_f, 1.0f);

        const float b = f_inf / τ;
        const float a = -1.0f / τ;

        return a*F + b;
    }

    inline float rhs_fdv(const float v, const float F) const
    {
        return exp((v-E_f)/k_f)/(k_f*τ_f*(exp((v-E_f)/k_f)+1)*(exp((v-E_f)/k_f)+1));
    }

    inline float rhs_fdf(const float v, const float F) const
    {
        const float τ     = τ_f;
        // const float f_inf = sigmoid(v, E_f, k_f, 1.0f);

        // const float b = f_inf / τ;
        const float a = -1.0f / τ;

        return a;
    }

    inline float update_f(const float v, const float F, const float Δt) const
    {
        const float τ     = τ_f;
        const float f_inf = sigmoid(v, E_f, k_f, 1.0f);

        const float b = f_inf / τ;
        const float a = -1.0f / τ;

        return std_clamp<float>(exp(a * Δt) * (F + b / a) - b / a, 0.0f,
                          1.0f);
    }

    inline float rhs_xr(const float v, const float Xr) const
    {
        const float τ      = τ_xr;
        const float xr_inf = sigmoid(v, E_xr, k_xr, -1.0f);

        const float b = xr_inf / τ;
        const float a = -1.0f / τ;

        return a*Xr + b;
    }

    inline float rhs_xrdv(const float v, const float Xr) const
    {
        return exp(-(v-E_xr)/k_xr)/(k_xr*τ_xr*(exp(-(v-E_xr)/k_xr)+1)*(exp(-(v-E_xr)/k_xr)+1));
    }

    inline float rhs_xrdxr(const float v, const float Xr) const
    {
        const float τ      = τ_xr;
        // const float xr_inf = sigmoid(v, E_xr, k_xr, -1.0f);

        // const float b = xr_inf / τ;
        const float a = -1.0f / τ;

        return a;
    }

    inline float update_xr(const float v, const float Xr, const float Δt) const
    {
        const float τ      = τ_xr;
        const float xr_inf = sigmoid(v, E_xr, k_xr, -1.0f);

        const float b = xr_inf / τ;
        const float a = -1.0f / τ;

        return std_clamp<float>(exp(a * Δt) * (Xr + b / a) - b / a, 0.0f,
                          1.0f);
    }

    inline float rhs_xs(const float v, const float Xs) const
    {
        const float τ      = τ_xs;
        const float xs_inf = sigmoid(v, E_xs, k_xs, -1.0f);

        const float b = xs_inf / τ;
        const float a = -1.0f / τ;

        return a*Xs + b;
    }

    inline float rhs_xsdv(const float v, const float Xs) const
    {
        return exp(-(v-E_xs)/k_xs)/(k_xs*τ_xs*(exp(-(v-E_xs)/k_xs)+1)*(exp(-(v-E_xs)/k_xs)+1));
    }

    inline float rhs_xsdxs(const float v, const float Xs) const
    {
        const float τ      = τ_xs;
        // const float xs_inf = sigmoid(v, E_xs, k_xs, -1.0f);

        // const float b = xs_inf / τ;
        const float a = -1.0f / τ;

        return a;
    }

    inline float update_xs(const float v, const float Xs, const float Δt) const
    {
        const float τ      = τ_xs;
        const float xs_inf = sigmoid(v, E_xs, k_xs, -1.0f);

        const float b = xs_inf / τ;
        const float a = -1.0f / τ;

        return std_clamp<float>(exp(a * Δt) * (Xs + b / a) - b / a, 0.0f,
                          1.0f);
    }

public:
    PathmanathanCordeiroGrayCellSolver(double C_m_)
    : C_m(C_m_)
    {
    }
    ~PathmanathanCordeiroGrayCellSolver() = default;

    void RescaleVoltage(mfem::GridFunction& φₘgf) const override
    {
        for(int i=0; i<φₘgf.Size(); i++) {
            double offset = 1-φₘgf(i);
            φₘgf(i) = -85.0*offset + (1.0-offset)*-5.0;
        }
    }

    int HGateIndex() const override { return 0; };

    void Init(mfem::GridFunction& Vgf, mfem::GridFunction& sgf) const override
    {
        assert(sgf.FESpace()->GetOrdering() == mfem::Ordering::byVDIM);

        for (int node = 0; node < Vgf.FESpace()->GetNDofs(); node++)
        {
            Vgf(node)    = E_K;
        }

        for (int node = 0; node < sgf.FESpace()->GetNDofs(); node++)
        {
            const auto V = E_K;

            sgf(6 * node + 0) = sigmoid(V, E_h, k_h, 1.0f);
            sgf(6 * node + 1) = sigmoid(V, E_m, k_m, -1.0f);
            sgf(6 * node + 2) = sigmoid(V, E_f, k_f, 1.0f);
            sgf(6 * node + 3) = sigmoid(V, E_s, k_s, 1.0f);
            sgf(6 * node + 4) = sigmoid(V, E_xs, k_xs, -1.0f);
            sgf(6 * node + 5) = sigmoid(V, E_xr, k_xr, -1.0f);
        }
    }

    void FullExplicitStep(mfem::Vector& cells_V, mfem::Vector& cells_s, float t, float Δt) const override
    {
        for (int i = 0; i < cells_V.Size(); i++)
        {
            float actual_Δt = Δt;

            const double V = cells_V(i);

            const float h  = cells_s(6 * i + 0);
            const float m  = cells_s(6 * i + 1);
            const float f  = cells_s(6 * i + 2);
            const float s  = cells_s(6 * i + 3);
            const float xs = cells_s(6 * i + 4);
            const float xr = cells_s(6 * i + 5);

            // Update gates
            const float h_  = update_h (V, h,  actual_Δt);
            const float m_  = update_m (V, m,  actual_Δt);
            const float f_  = update_f (V, f,  actual_Δt);
            const float s_  = update_s (V, s,  actual_Δt);
            const float xs_ = update_xs(V, xs, actual_Δt);
            const float xr_ = update_xr(V, xr, actual_Δt);

            // Instantaneous gates
            const float r = sigmoid(V, E_r, k_r, -1.0);
            const float d = sigmoid(V, E_d, k_d, -1.0);
            const float z = sigmoid(V, E_z, k_z, 1.0);
            const float y = sigmoid(V, E_y, k_y, 1.0);

            // Currents
            const float I_Na  = g_Na * m * m * m * h * h * (V - E_Na);
            const float I_K1  = g_K1 * z * (V - E_K);
            const float I_to  = g_to * r * s * (V - E_K);
            const float I_CaL = g_CaL * d * f * (V - E_Ca);
            const float I_Kr  = g_Kr * xr * y * (V - E_K);
            const float I_Ks  = g_Ks * xs * (V - E_K);

            const float I_total =
                I_Na + I_K1 + I_to + I_CaL + I_Kr + I_Ks;

            // Actual step
            cells_V(i) = V - actual_Δt * I_total / C_m;

            cells_s(6 * i + 0) = h_;
            cells_s(6 * i + 1) = m_;
            cells_s(6 * i + 2) = f_;
            cells_s(6 * i + 3) = s_;
            cells_s(6 * i + 4) = xs_;
            cells_s(6 * i + 5) = xr_;
        }
    }

    //   //! NOTE: int_point must fit to cells_s.
    double EvalReaction(const double V, const mfem::Vector &cells_s, int int_point, float t) const override
    {
        const float h = cells_s(6 * int_point + 0);
        const float m = cells_s(6 * int_point + 1);
        const float f = cells_s(6 * int_point + 2);
        const float s = cells_s(6 * int_point + 3);
        const float xs = cells_s(6 * int_point + 4);
        const float xr = cells_s(6 * int_point + 5);

        // Instantaneous gates
        const float r = sigmoid(V, E_r, k_r, -1.0);
        const float d = sigmoid(V, E_d, k_d, -1.0);
        const float z = sigmoid(V, E_z, k_z, 1.0);
        const float y = sigmoid(V, E_y, k_y, 1.0);
            
        // Currents
        const float I_Na = g_Na * m * m * m * h * h * (V - E_Na);
        const float I_K1 = g_K1 * z * (V - E_K);
        const float I_to = g_to * r * s * (V - E_K);
        const float I_CaL = g_CaL * d * f * (V - E_Ca);
        const float I_Kr = g_Kr * xr * y * (V - E_K);
        const float I_Ks = g_Ks * xs * (V - E_K);

        return I_Na + I_K1 + I_to + I_CaL + I_Kr + I_Ks;
    }

    void InternalRHS(mfem::Vector&rhs, double V, mfem::Vector &cells_s) const override
    {
        // h
        rhs(0) = rhs_h(V, cells_s(0));
        // m
        rhs(1) = rhs_m(V, cells_s(1));
        // f
        rhs(2) = rhs_f(V, cells_s(2));
        // s
        rhs(3) = rhs_s(V, cells_s(3));
        // xs
        rhs(4) = rhs_xs(V, cells_s(4));
        // xr
        rhs(5) = rhs_xr(V, cells_s(5));
    }

    void InternalEulerUpdate(const double V, mfem::Vector &cells_s, int int_point, float t, double Δt) const override
    {
        cells_s(6 * int_point + 0) += Δt*rhs_h (V, cells_s(6 * int_point + 0));
        cells_s(6 * int_point + 1) += Δt*rhs_m (V, cells_s(6 * int_point + 1));
        cells_s(6 * int_point + 2) += Δt*rhs_f (V, cells_s(6 * int_point + 2));
        cells_s(6 * int_point + 3) += Δt*rhs_s (V, cells_s(6 * int_point + 3));
        cells_s(6 * int_point + 4) += Δt*rhs_xs(V, cells_s(6 * int_point + 4));
        cells_s(6 * int_point + 5) += Δt*rhs_xr(V, cells_s(6 * int_point + 5));

        for(int i=0;i<6;i++) {
            cells_s(6 * int_point + i) = std_clamp<float>(cells_s(6 * int_point + i), 0.0f, 1.0f);
        }
    }

    int InternalDim() const override {return 6;}
};
