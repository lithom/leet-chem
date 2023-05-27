package tech.molecules.analytics.activitycliff;

public class FoldChangeActivityCliffDefinition implements ActivityCliffDefinition {
    private double a; // threshold for individual value
    private double b; // fold-change threshold
    private double c; // absolute difference threshold

    public FoldChangeActivityCliffDefinition(double a_ub, double b_minfoldchange, double c_minabschange) {
        this.a = a_ub;
        this.b = b_minfoldchange;
        this.c = c_minabschange;
    }

    @Override
    public int computeActivityCliff(double v1, double v2) {
        // Check for at least one value below threshold a
        if (v1 > a && v2 > a) return 0;

        // Check for fold-change above threshold b
        if (v1 / v2 < b && v2 / v1 < b) return 0;

        // Check for absolute difference above threshold c
        if (Math.abs(v1 - v2) < c) return 0;

        if( v1 > v2) {return 1;}
        return -1;
    }
}
