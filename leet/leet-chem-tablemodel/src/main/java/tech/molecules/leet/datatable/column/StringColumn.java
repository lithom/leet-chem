package tech.molecules.leet.datatable.column;

public class StringColumn extends AbstractDataTableColumn<String,String> {

    public StringColumn() {
        super(String.class);
    }
    @Override
    public String processData(String data) {
        return data;
    }

}
