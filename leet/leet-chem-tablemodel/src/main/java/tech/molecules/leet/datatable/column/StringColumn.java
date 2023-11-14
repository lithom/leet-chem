package tech.molecules.leet.datatable.column;

import java.io.Serializable;

public class StringColumn extends AbstractDataTableColumn<String,String> implements Serializable {

    // Explicit serialVersionUID for interoperability
    private static final long serialVersionUID = 1L;

    public StringColumn() {
        super(String.class);
    }
    @Override
    public String processData(String data) {
        return data;
    }

}
