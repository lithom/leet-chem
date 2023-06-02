package tech.molecules.analytics;

public class MMPQuery {

    public static class AssayValue {
        public final int assayId;
        public final String property;
        public AssayValue(int assayId, String property) {
            this.assayId = assayId;
            this.property = property;
        }
    }

    private AssayValue assayValue = null;

    private String synthon = null;

    private int minMMPs = -1;

    public MMPQuery(AssayValue assayValue, String synthon, int minMMPs) {
        this.assayValue = assayValue;
        this.synthon = synthon;
        this.minMMPs = minMMPs;
    }

    public AssayValue getAssayValue() {
        return assayValue;
    }

    public void setAssayValue(AssayValue assayValue) {
        this.assayValue = assayValue;
    }

    public String getSynthon() {
        return synthon;
    }

    public void setSynthon(String synthon) {
        this.synthon = synthon;
    }

    public int getMinMMPs() {
        return minMMPs;
    }

    public void setMinMMPs(int minMMPs) {
        this.minMMPs = minMMPs;
    }
}
