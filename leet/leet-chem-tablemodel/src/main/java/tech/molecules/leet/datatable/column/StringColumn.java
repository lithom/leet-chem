package tech.molecules.leet.datatable.column;

public class StringColumn extends AbstractDataTableColumn<String,String> {

    @Override
    public String processData(String data) {
        return data;
    }

}
