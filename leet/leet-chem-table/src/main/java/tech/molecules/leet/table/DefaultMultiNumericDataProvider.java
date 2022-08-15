package tech.molecules.leet.table;

import java.util.Map;

public class DefaultMultiNumericDataProvider implements NDataProvider.NMultiNumericDataProvider {

    private Map<String,double[]> data;

    public DefaultMultiNumericDataProvider(Map<String,double[]> data) {
        this.data = data;
    }

    @Override
    public double[] getMultiNumericData(String ki) {
        return this.data.get(ki);
    }

}
