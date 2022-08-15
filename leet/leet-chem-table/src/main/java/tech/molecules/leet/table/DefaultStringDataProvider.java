package tech.molecules.leet.table;

import java.util.Map;

public class DefaultStringDataProvider implements NDataProvider.NStringDataProvider {

    private Map<String,String> data;

    public DefaultStringDataProvider(Map<String,String> data) {
        this.data = data;
    }

    @Override
    public String getStringData(String rowid) {
        return this.data.get(rowid);
    }
}
