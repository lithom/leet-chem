package tech.molecules.leet.chem.sar.analysis;

import java.util.List;

public interface MatchedSeries {

    public static interface MatchedSeriesElement {
        public String getStructure();
        public Part getVariant();
        public Part getCommonRemainder();
    }
    public Part getCommonRemainder();
    public List<MatchedSeriesElement> getSeries();
}
