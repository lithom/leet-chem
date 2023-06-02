package tech.molecules.analytics;

public class MMPNumericQuery extends MMPQuery {

    private Double minRatioUp   = null;
    private Double minRatioDown = null;

    public MMPNumericQuery(AssayValue assayValue, String synthon, int minMMPs, Double minRatioUp, Double minRatioDown) {
        super(assayValue, synthon, minMMPs);
        this.minRatioUp = minRatioUp;
        this.minRatioDown = minRatioDown;
    }

}
