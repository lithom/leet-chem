package tech.molecules.analytics.activitycliff;

public interface ActivityCliffDefinition {

    /**
     *
     * @return 0 for no activity cliff, 1 for a high, b low, -1 for a low, b high
     */
    public int computeActivityCliff(double v1, double v2);

    public default boolean isActivityCliff(double v1, double v2) {
        return computeActivityCliff(v1,v2)!=0;
    }
}
