package tech.molecules.leet.datatable.microleet;

import com.formdev.flatlaf.FlatLightLaf;
import javafx.scene.chart.Chart;
import net.mahdilamb.colormap.Colormap;
import net.mahdilamb.colormap.Colormaps;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.DefaultXYItemRenderer;
import org.jfree.data.json.impl.JSONArray;
import tech.molecules.leet.datatable.*;
import tech.molecules.leet.datatable.chart.jfc.LeetXYZDataSet;
import tech.molecules.leet.datatable.chart.jfc.VisualizationComponent;
import tech.molecules.leet.datatable.chem.CalculatedBasicStructurePropertyNumericDatasource;
import tech.molecules.leet.datatable.microleet.importer.DataWarriorTSVParserHelper;
import tech.molecules.leet.datatable.microleet.model.MicroLeetDataModel;
import tech.molecules.leet.datatable.microleet.model.MicroLeetDataType;
import tech.molecules.leet.datatable.microleet.task.AddPlotAction;
import tech.molecules.leet.datatable.microleet.task.ImportCSVAction;
import tech.molecules.leet.datatable.microleet.view.MicroLeetMainPanel;
import tech.molecules.leet.datatable.swing.*;
import tech.molecules.leet.datatable.swing.chem.FilterActionProviderSubstructure;
import tech.molecules.leet.gui.UtilSwing;
import tech.molecules.leet.io.CSVIterator;

import javax.swing.*;
import javax.swing.table.TableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

public class MicroLeetMain {

    public static MicroLeetMain Main;
    public static MicroLeetDataModel DataModel;

    private MicroLeetDataModel dataModel;



    private JFrame mainFrame;
    private JMenuBar mainMenu;
    private JMenu menuFile;
    private JMenu menuChem;

    // layout
    private MicroLeetMainPanel pMain;
    private DefaultSwingTableController swingTableController;






    public MicroLeetMain() {
        if( MicroLeetMain.Main == null) {
            MicroLeetMain.Main = this;
        }
    }



    public void init() {
        dataModel = new MicroLeetDataModel();
        MicroLeetMain.DataModel = dataModel;

        // GUI
        reinitFrame();
        reinitSwingTableController();
        //reinitLayout();
        reinitMenu();
    }

    private void reinitFrame() {
        FlatLightLaf.setup();
        try {
            UIManager.setLookAndFeel(new FlatLightLaf());
        } catch (Exception ex) {
            System.err.println("Failed to initialize LaF");
        }

        this.mainFrame = new JFrame();
        mainFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        //mainFrame.getContentPane();
        mainFrame.setSize(1024,800);

        mainFrame.setVisible(true);
    }

    private void reinitLayout() {
        this.mainFrame.getContentPane().removeAll();
        this.mainFrame.setLayout(new BorderLayout());

        //this.pMain = new MicroLeetMainPanel(this);
        //this.pMain.add(this.pMain,BorderLayout.CENTER);
        //this.pMainTable = new JPanel();
        //this.mainFrame.add(this.pMainTable,BorderLayout.CENTER);
    }

    private Supplier<JPanel> getFilterPanelSupplier() {
        Supplier<JPanel> sp = new Supplier<JPanel>() {
            @Override
            public JPanel get() {
                return pMain.createAdditionalFilterPanel();
            }
        };
        return sp;
    }

    private Supplier<JPanel> getPlotPanelSupplier() {
        Supplier<JPanel> sp = new Supplier<JPanel>() {
            @Override
            public JPanel get() {
                return pMain.createAdditionalPlotPanel();
            }
        };
        return sp;
    }

    public Action getNewPlotAction() {
        Action newPlotAction = new AddPlotAction(Main.dataModel.getDataTable(), () -> pMain.createAdditionalPlotPanel() );
        return newPlotAction;
    }

    private List<FilterActionProvider> filterActionProviders;

    private void initFilterActionProviders() {
        this.filterActionProviders = new ArrayList<>();
        this.filterActionProviders.add(new FilterActionProviderSubstructure(getFilterPanelSupplier()));

        //this.filterActionProviders.add(new FilterActionProviderSubstructure(new UtilSwing.PanelAsFrameProvider(this.pMain,100,100)));
    }

    private void reinitSwingTableController() {

        initFilterActionProviders();

        DefaultSwingTableModel tm = this.dataModel.getSwingTableModel();
        this.swingTableController = new DefaultSwingTableController(tm,filterActionProviders,getFilterPanelSupplier());
        this.pMain = new MicroLeetMainPanel(this);
        this.mainFrame.getContentPane().removeAll();
        this.mainFrame.getContentPane().setLayout(new BorderLayout());
        this.mainFrame.getContentPane().add(this.pMain,BorderLayout.CENTER);

        // create FilterActionResolvers
        //this.pMainTable.removeAll();
        //this.pMainTable.setLayout(new BorderLayout());
        //this.pMainTable.add(this.swingTableController,BorderLayout.CENTER);
    }

    private void reinitMenu() {
        this.mainMenu = new JMenuBar();
        this.mainFrame.setJMenuBar(this.mainMenu);

        this.menuFile = new JMenu("File");
        this.menuChem = new JMenu("Chemistry");

        this.mainMenu.add(this.menuFile);
        this.mainMenu.add(this.menuChem);


        this.menuFile.add( new AbstractAction("TestSerialization"){
            @Override
            public void actionPerformed(ActionEvent e) {
                DataTableSerializer dts = new DataTableSerializer();
                dts.serialize( dataModel.getDataTable() );
                DataTable dt2 = dts.deserializeDataTable( "datatable1.data" );
                System.out.println("deserialized..");
            }
        });

    }


    public static class FunctionA implements Function<Double, Color>, Serializable {
        transient Colormap colormap_0 = Colormaps.Sequential.CubeYF();
        @Override
        public Color apply(Double aDouble) {
            Color ca = colormap_0.get( (float) Math.min(1.0,(aDouble/2000.0)));
            return new Color(ca.getRed(),ca.getGreen(),ca.getBlue(),60);
        }
        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            in.defaultReadObject();
            this.colormap_0 = Colormaps.Sequential.CubeYF();
        }
    };

    public static void main(String args[]) {
        MicroLeetMain main = new MicroLeetMain();
        main.init();

        //String filepath = "C:\\Users\\liphath1\\OneDrive - Idorsia\\Documents\\Osiris_Project_Profile_TEST.tsv";
        //String filepath = "C:\\datasets\\chembl_size90_input_smiles_SERIES_A_1.txt";
        String filepath = "/home/liphath1/datasets/ListOfSmallMoleculeDrugsFrlomDrugCentral2021.txt";
        // load test dwar tsv file..
        if(true) {
            DataWarriorTSVParserHelper helper_a = new DataWarriorTSVParserHelper();
            try {
                List<String> colTypes = helper_a.getColumnTypesDefault(filepath);
                Map<Integer,MicroLeetDataType> columns = new HashMap<>();
                Map<Integer,String> colNames = new HashMap<>();
                int cnt = 0;
                List<Integer> chem_cols = new ArrayList<>();
                List<Integer> numeric_cols = new ArrayList<>();
                List<Integer> cols_to_parse = new ArrayList<>();
                for(String ci : colTypes) {
                    if (ci.equals("chem_structure")) {
                        columns.put(cnt, DataModel.getDataTypeChemStructure());
                        chem_cols.add(cnt);
                    } else if (ci.equals("multinumeric")) {
                        columns.put(cnt, DataModel.getDataTypeMultiNumeric());
                        numeric_cols.add(cnt);
                    } else {
                        columns.put(cnt, DataModel.getDataTypeString());
                    }
                    colNames.put(cnt,"Col_"+cnt);
                    cols_to_parse.add(cnt);
                    cnt++;
                }

                CSVIterator lineIterator = new CSVIterator(new BufferedReader(new FileReader(filepath)),true,cols_to_parse,"\\t");
                DataModel.loadCSVFile(lineIterator,1,columns,colNames);

                // Color the activity column a bit
                DataTableColumn col_0 = main.getSwingTableController().getModel().getDataTable().getDataColumns().get(numeric_cols.get(0));
                FunctionA f_0 = new FunctionA();
                col_0.setBackgroundColor((NumericDatasource) col_0.getNumericDatasources().get(0),f_0);


                CellRendererHelper.configureDefaultRenderers(MicroLeetMain.Main.getSwingTableController());
                MicroLeetMain.Main.getSwingTableController().setRowHeight(120);
                System.out.println("loaded!");



                JFrame f2 = new JFrame();
                f2.getContentPane().setLayout(new BorderLayout());
                ActiveFilterController afc = new ActiveFilterController(MicroLeetMain.Main.getSwingTableController());
                f2.getContentPane().add(afc,BorderLayout.CENTER);
                f2.setSize(new Dimension(600,600));
                f2.setVisible(true);


                // Create Plot..
                DataTable table = MicroLeetMain.Main.dataModel.getDataTable();
                DataTableSerializer dts = new DataTableSerializer();
                dts.serialize(table);
                System.out.println("mkay");

                NumericDatasource ndx = (NumericDatasource) table.getDataColumns().get( numeric_cols.get(0) ).getNumericDatasources().get(0);
                NumericDatasource ndy = (NumericDatasource) table.getDataColumns().get( numeric_cols.get(1) ).getNumericDatasources().get(0);
                LeetXYZDataSet dataset = new LeetXYZDataSet(MicroLeetMain.Main.dataModel.getDataTable());
                dataset.setDataSources(ndx,ndy,null,null);
                LeetXYZDataSet dataset_b = new LeetXYZDataSet(MicroLeetMain.Main.dataModel.getDataTable());
                dataset_b.setDataSources(new CalculatedBasicStructurePropertyNumericDatasource(table.getDataColumns().get(chem_cols.get(0)),1),ndy,null,null);
                LeetXYZDataSet dataset_c = new LeetXYZDataSet(MicroLeetMain.Main.dataModel.getDataTable());
                dataset_c.setDataSources(new CalculatedBasicStructurePropertyNumericDatasource(table.getDataColumns().get(chem_cols.get(0)),3),new CalculatedBasicStructurePropertyNumericDatasource(table.getDataColumns().get(chem_cols.get(0)),6),null,null);
                LeetXYZDataSet dataset_d = new LeetXYZDataSet(MicroLeetMain.Main.dataModel.getDataTable());
                dataset_d.setDataSources(ndx,new CalculatedBasicStructurePropertyNumericDatasource(table.getDataColumns().get(chem_cols.get(0)),10),null,null);

                JFrame f4 = new JFrame();
                VisualizationComponent viscomp = new VisualizationComponent(table);
                f4.getContentPane().setLayout(new BorderLayout());
                f4.getContentPane().add(viscomp);
                f4.setSize(new Dimension(600,600));
                f4.setVisible(true);

                JFrame f3 = new JFrame();
                f3.getContentPane().setLayout(new GridLayout(2,2));
                DefaultXYItemRenderer renderer = new DefaultXYItemRenderer();
                renderer.setDefaultLinesVisible(false);
                XYPlot plot_a = new XYPlot(dataset,new NumberAxis("x"),new NumberAxis("y"),renderer);
                JFreeChart chart = new JFreeChart(plot_a);
                f3.getContentPane().add(new ChartPanel(chart),BorderLayout.CENTER);
                XYPlot plot_b = new XYPlot(dataset_b,new NumberAxis("x"),new NumberAxis("y"),renderer);
                JFreeChart chart_b = new JFreeChart(plot_b);
                f3.getContentPane().add(new ChartPanel(chart_b),BorderLayout.CENTER);
                XYPlot plot_c = new XYPlot(dataset_c,new NumberAxis("x"),new NumberAxis("y"),renderer);
                JFreeChart chart_c = new JFreeChart(plot_c);
                f3.getContentPane().add(new ChartPanel(chart_c),BorderLayout.CENTER);
                XYPlot plot_d = new XYPlot(dataset_d,new NumberAxis("x"),new NumberAxis("y"),renderer);
                JFreeChart chart_d = new JFreeChart(plot_d);
                f3.getContentPane().add(new ChartPanel(chart_d),BorderLayout.CENTER);


                f3.setSize(new Dimension(600,600));
                f3.setVisible(true);


            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public DefaultSwingTableController getSwingTableController() {
        return this.swingTableController;
    }

}
