package tech.molecules.leet.datatable.swing.testA;

import tech.molecules.leet.datatable.*;
import tech.molecules.leet.datatable.column.AbstractDataTableColumn;
import tech.molecules.leet.datatable.dataprovider.HashMapBasedDataProvider;
import tech.molecules.leet.datatable.swing.DefaultSwingTableController;
import tech.molecules.leet.datatable.swing.DefaultSwingTableModel;
import tech.molecules.leet.datatable.swing2.InteractiveTableModel;

import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;

public class TestSwingTable_A {


    public static class RandomStringDataProvider extends HashMapBasedDataProvider<String> {
        Map<String,String> entries = new HashMap();
        public RandomStringDataProvider(int n) {
            super(new HashMap<>());
            InteractiveTableModel.RandomStringGenerator rg = new InteractiveTableModel.RandomStringGenerator();
            for(int zi=0;zi<n;zi++) {
                entries.put("id"+zi,rg.generateRandomString(12));
            }
            this.addData(entries);
        }
    }

    public static class StringColumn extends AbstractDataTableColumn<String,String> {
        @Override
        public String processData(String data) {
            return data;
        }

        @Override
        public List<NumericDatasource> getNumericDatasources() {
            List<NumericDatasource> ds = new ArrayList<>();
            ds.add(new AbstractNumericDatasource<String>("StringLength",getThisColumn()) {
                @Override
                public Double evaluate(String original) {
                    if(original==null) {return Double.NaN;}
                    return (double) original.length();
                }
            });
            return ds;
        }
    }

    DataProvider<String> dp_a = new RandomStringDataProvider(2000);

    public static void main(String args[]) {

        DataProvider<String> dp_a = new RandomStringDataProvider(2000);

        DataTable dtable = new DataTable();
        DataTableColumn<String,String> dtc = new StringColumn();
        dtc.setDataProvider(dp_a);
        dtable.addDataColumn(dtc);

        JFrame fi = new JFrame();
        fi.getContentPane().setLayout(new BorderLayout());

        DefaultSwingTableModel swingmodel = new DefaultSwingTableModel(dtable);
        DefaultSwingTableController table = new DefaultSwingTableController(swingmodel);

        dtable.setAllKeys(dp_a.getAllEntries());

        fi.getContentPane().add(table);
        fi.setSize(600,600);
        fi.setVisible(true);
    }


}
