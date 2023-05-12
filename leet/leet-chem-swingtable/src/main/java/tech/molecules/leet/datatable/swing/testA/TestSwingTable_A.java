package tech.molecules.leet.datatable.swing.testA;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.table.ChemistryCellRenderer;
import com.formdev.flatlaf.FlatLightLaf;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.chem.mutator.properties.ChemPropertyCounts;
import tech.molecules.leet.datatable.*;
import tech.molecules.leet.datatable.chem.DefaultStructureProvider;
import tech.molecules.leet.datatable.chem.StructurePropertiesColumn;
import tech.molecules.leet.datatable.chem.StructurePropertiesProvider;
import tech.molecules.leet.datatable.chem.StructureWithIDColumn;
import tech.molecules.leet.datatable.column.AbstractDataTableColumn;
import tech.molecules.leet.datatable.dataprovider.HashMapBasedDataProvider;
import tech.molecules.leet.datatable.filter.StringRegExpFilter;
import tech.molecules.leet.datatable.swing.DefaultSwingTableController;
import tech.molecules.leet.datatable.swing.DefaultSwingTableModel;
import tech.molecules.leet.datatable.swing.GridOfColoredStringsRenderer;
import tech.molecules.leet.datatable.swing2.InteractiveTableModel;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
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

    public static class StructureDataProvider extends DefaultStructureProvider {
        public StructureDataProvider(List<String> keys) {
            List<StructureWithID> sids = new ArrayList<>();
            List<StereoMolecule> mols = ChemUtils.loadTestMolecules_35FromDrugCentral();
            int zi=0;
            for(String ki : keys) {
                sids.add(new StructureWithID(ki,"",mols.get(zi%35)));
                zi++;
            }
            this.loadStructures(sids);
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

        FlatLightLaf.setup();
        try {
            UIManager.setLookAndFeel( new FlatLightLaf() );
        } catch( Exception ex ) {
            System.err.println( "Failed to initialize LaF" );
        }

        int n_rows = 200000;
        int ms_break = 2500;
        int n_spcColumns = 5;

        DataProvider<String> dp_a = new RandomStringDataProvider(n_rows);
        DataProvider<StructureWithID> dp_b = new StructureDataProvider(dp_a.getAllEntries());
        DataProvider<StructurePropertiesProvider.CalculatedBasicStructureProperties> dp_c = new StructurePropertiesProvider(dp_b);

        DataTable dtable = new DataTable();

        DataTableColumn<String,String> dtc = new StringColumn();
        dtc.setDataProvider(dp_a);
        dtable.addDataColumn(dtc);

        StructureWithIDColumn sic = new StructureWithIDColumn();
        sic.setDataProvider(dp_b);
        dtable.addDataColumn(sic);

        List<StructurePropertiesColumn> spcs = new ArrayList<>();
        for(int zi=0;zi<n_spcColumns;zi++) {
            StructurePropertiesColumn spci = new StructurePropertiesColumn();
            spci.setDataProvider(dp_c);
            dtable.addDataColumn(spci);
            spcs.add(spci);
        }

        JFrame fi = new JFrame();
        fi.getContentPane().setLayout(new BorderLayout());

        DefaultSwingTableModel swingmodel = new DefaultSwingTableModel(dtable);
        DefaultSwingTableController table = new DefaultSwingTableController(swingmodel);

        dtable.setAllKeys(dp_a.getAllEntries());

        NumericDatasource nd = (NumericDatasource) dtable.getDataColumns().get(0).getNumericDatasources().get(0);
        dtable.getDataColumns().get(0).setBackgroundColor( nd , (x) -> new Color(255-3*(int)x,255,255) );
        table.setTableCellRenderer(0,new DefaultTableCellRenderer(){
            @Override
            public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
                if(value instanceof DataTableColumn.CellValue) {
                    DataTableColumn.CellValue cv = (DataTableColumn.CellValue) value;
                    this.setText(cv.val.toString());
                }
                return this;
            }
        });

        table.setTableCellRenderer(1,new ChemistryCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int col) {
                String[] val_a = null;
                if(value instanceof DataTableColumn.CellValue) {
                    DataTableColumn.CellValue value_cv = (DataTableColumn.CellValue) value;
                    if( value_cv.val instanceof StructureWithID) {
                        val_a = ((StructureWithID) value_cv.val).structure;
                    }
                }
                return super.getTableCellRendererComponent(table, val_a[0]+" "+val_a[1], isSelected, hasFocus, row, col);
            }
        });

        if(false) {
            table.setTableCellRenderer(2, new DefaultTableCellRenderer() {
                @Override
                public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
                    if (value instanceof DataTableColumn.CellValue) {
                        DataTableColumn.CellValue value_cv = (DataTableColumn.CellValue) value;
                        if (value_cv.val instanceof StructurePropertiesProvider.CalculatedBasicStructureProperties) {
                            StructurePropertiesProvider.CalculatedBasicStructureProperties cbsp = (StructurePropertiesProvider.CalculatedBasicStructureProperties) value_cv.val;
                            StringBuilder sba = new StringBuilder();
                            for (int zi = 0; zi < ChemPropertyCounts.COUNTS_ALL.length; zi++) {
                                sba.append(zi + "->" + cbsp.counts.get(zi) + ";");
                                if (zi % 4 == 0) {
                                    sba.append("\n");
                                }
                            }
                            this.setText(sba.toString());
                        }
                    }
                    return this;
                }
            });
        }

        for(int zc=2;zc<dtable.getDataColumns().size();zc++) {
            table.setTableCellRenderer(zc, new GridOfColoredStringsRenderer() {
                @Override
                public GridOfColoredStrings convertToGridOfColoredStrings(Object obj) {
                    ColoredString strings[] = new ColoredString[ChemPropertyCounts.COUNTS_ALL.length];
                    if (obj instanceof DataTableColumn.CellValue) {
                        DataTableColumn.CellValue value_cv = (DataTableColumn.CellValue) obj;
                        if (value_cv.val instanceof StructurePropertiesProvider.CalculatedBasicStructureProperties) {
                            StructurePropertiesProvider.CalculatedBasicStructureProperties cbsp = (StructurePropertiesProvider.CalculatedBasicStructureProperties) value_cv.val;
                            for (int zi = 0; zi < ChemPropertyCounts.COUNTS_ALL.length; zi++) {
                                String si = (zi + "->" + cbsp.counts.get(zi) + ";");
                                strings[zi] = new ColoredString(si, new Color(255 - cbsp.counts.get(zi), 255, 255));
                            }
                            //this.setText(sba.toString());
                        }
                    }
                    return new GridOfColoredStrings(4, 4, strings);
                }
            });
        }


        table.setRowHeight(120);


        fi.getContentPane().add(table);
        fi.setSize(600,600);
        fi.setVisible(true);


        try {
            Thread.sleep(ms_break);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        StringRegExpFilter filter_a = new StringRegExpFilter();
        filter_a.setRegExp(".*aa.*");
        dtable.addFilter(dtable.getDataColumns().get(0),filter_a);
        System.out.println("Filter set");

        try {
            Thread.sleep(ms_break);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        DataSort<String> sort1 = new DataSort<String>() {
            @Override
            public String getName() {
                return "Sort1";
            }
            @Override
            public int compare(String a, String b) {
                return Character.compare(a.charAt(3),b.charAt(3));
            }
        };

        DataSort<String> sort2 = new DataSort<String>() {
            @Override
            public String getName() {
                return "Sort2";
            }
            @Override
            public int compare(String a, String b) {
                return Character.compare(a.charAt(5),b.charAt(5));
            }
        };
        List<Pair<DataTableColumn,DataSort>> sorts = new ArrayList<>();
        sorts.add(Pair.of(dtable.getDataColumns().get(0), sort1));
        sorts.add(Pair.of(dtable.getDataColumns().get(0), sort2));

        dtable.setDataSort(sorts);

    }


}
