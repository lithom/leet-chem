package tech.molecules.leet.gui.chem.echelon.view;

import com.actelion.research.gui.table.ChemistryCellRenderer;
import tech.molecules.leet.chem.IOUtils;
import tech.molecules.leet.chem.sar.SimpleMultiSynthonStructure;
import tech.molecules.leet.chem.sar.SimpleSARProject;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.chem.sar.tree.SARProjectTreeNode;
import tech.molecules.leet.gui.chem.echelon.action.ShowSinglePartAnalysisAction;
import tech.molecules.leet.gui.chem.echelon.model.EchelonModel;
import tech.molecules.leet.gui.chem.project.ProjectTreeController;
import tech.molecules.leet.gui.chem.project.SeriesManagerTreeView;
import tech.molecules.leet.gui.chem.project.action.AddSeriesAction;
import tech.molecules.leet.gui.chem.project.action.EditMultiSynthonStructureAction;
import tech.molecules.leet.gui.chem.project.action.LoadProjectAction;
import tech.molecules.leet.gui.chem.project.action.SaveProjectAction;

import javax.swing.*;
import javax.swing.tree.DefaultTreeModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.*;
import java.util.List;

public class EchelonViewMain extends JPanel {

    private EchelonModel model;

    private ProjectTreeController projectTreeController;

    private JTabbedPane seriesTabbedPane;

    private SeriesManagerTreeView seriesManagerTreeView;
    private JTable seriesTable;
    private JTable compoundsTable;
    private JTextField seriesDefinitionField;

    private JMenuBar menuBar;

    public EchelonViewMain(EchelonModel model) {
        this.model = model;
        reinit();
    }

    private void initProjectTreeController() {
        this.projectTreeController = new ProjectTreeController();
        ProjectTreeController tc = this.projectTreeController;
        //SimpleSARDecompositionModel model = new SimpleSARDecompositionModel(new ArrayList<>());
        SimpleSARProject project = model.getDecompositionModel().getProject();
        SARProjectTreeNode projectRootNode = new SARProjectTreeNode(project, null);
        DefaultTreeModel tm = new DefaultTreeModel(projectRootNode);

        List<Class<?>> checkedClasses = new ArrayList<>();
        checkedClasses.add(SimpleSARSeries.class);
        checkedClasses.add(SimpleSARProject.class);
        List<Action> sarProjectActions = new ArrayList<>();
        sarProjectActions.add(new LoadProjectAction((pi) -> tm.setRoot(new SARProjectTreeNode(pi, null))));
        sarProjectActions.add(new SaveProjectAction());
        sarProjectActions.add(new AddSeriesAction((xi) -> {
            project.getSeries().add(xi);
            tm.reload(projectRootNode);
        }));
        List<Action> seriesActions = new ArrayList<>();
        seriesActions.add(new EditMultiSynthonStructureAction((x) -> {
            SimpleSARSeries sis = new SimpleSARSeries("test", new SimpleMultiSynthonStructure(x));
            model.getDecompositionModel().setSeriesAndCompounds( Collections.singletonList(sis) , model.getCompounds() );
        }
        , () -> {
            JFrame f2 = new JFrame();
            f2.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            f2.getContentPane().setLayout(new BorderLayout());
            JPanel pi = new JPanel();
            pi.setLayout(new BorderLayout());
            f2.getContentPane().add(pi, BorderLayout.CENTER);
            f2.setSize(600, 600);
            f2.setVisible(true);
            return pi;
        }));

        tc.setListOfCheckedClasses(checkedClasses);
        tc.setNodeTypeActions(SimpleSARProject.class, sarProjectActions);
        tc.setNodeTypeActions(SimpleSARSeries.class, seriesActions);

        this.seriesManagerTreeView = new SeriesManagerTreeView(tm, tc);
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        // Left Panel
        JPanel leftPanel = new JPanel(new GridLayout(2, 1));

        // left tabbed pane
        this.seriesTabbedPane = new JTabbedPane();
        leftPanel.add(seriesTabbedPane);

        // Series Tree View
        initProjectTreeController();
        seriesTabbedPane.addTab("Series Configuration", seriesManagerTreeView);

        // Series Table
        seriesTable = new JTable( model.getSarTableModel() ); // You will need a custom table model
        JScrollPane seriesScrollPane = new JScrollPane(seriesTable);
        this.seriesTabbedPane.addTab("Table",seriesScrollPane);

        // Series Definition Configuration
        JPanel seriesConfigPanel = new JPanel(); // Use default FlowLayout
        seriesConfigPanel.add(new JLabel("Series Definition:"));
        seriesDefinitionField = new JTextField(20);
        seriesConfigPanel.add(seriesDefinitionField);
        JButton addSeriesButton = new JButton("Add Series");
        addSeriesButton.addActionListener(e -> fireCreateSeriesEvent());
        seriesConfigPanel.add(addSeriesButton);

        leftPanel.add(seriesConfigPanel);

        add(leftPanel, BorderLayout.WEST);

        // Right Panel (Compounds Table)
        compoundsTable = new JTable(model.getCompoundTableModel()); // You will need a custom table model
        // configure compounds view table..
        compoundsTable.getColumnModel().getColumn(0).setCellRenderer(new ChemistryCellRenderer());
        compoundsTable.getColumnModel().getColumn(0).setCellRenderer(new ChemistryCellRenderer());
        compoundsTable.getColumnModel().getColumn(2).setCellRenderer(new ChemistryCellRenderer());
        compoundsTable.getColumnModel().getColumn(3).setCellRenderer(new ChemistryCellRenderer());
        compoundsTable.getColumnModel().getColumn(4).setCellRenderer(new ChemistryCellRenderer());
        compoundsTable.setRowHeight(120);

        JScrollPane compoundsScrollPane = new JScrollPane(compoundsTable);
        add(compoundsScrollPane, BorderLayout.CENTER);

        this.menuBar = new JMenuBar();
        this.add(menuBar,BorderLayout.NORTH);

        initMenu();
    }

    //@Override
    //public void updateCompoundsTable(List<Compound> compounds) {
        // Update compounds table using the compounds list
        // This will likely involve creating a custom table model
    //}

    //@Override
    //public void updateSeriesTable(List<Series> seriesList) {
        // Update series table using the seriesList
        // This will likely involve creating a custom table model
    //}

    // ... Implement other interface methods

    private void fireCreateSeriesEvent() {
        // Trigger event for creating series (this will be linked to the controller)
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame fa = new JFrame();
            fa.setTitle("Echelon");
            fa.setSize(800, 600);
            fa.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            EchelonModel model = new EchelonModel();

            if(true) {
                //List<String> idcodes_a = IOUtils.extractSpecialColumnFromDWAR("C:\\datasets\\test_set_idotdo_a.dwar","Structure");
                List<String> idcodes_a = IOUtils.extractSpecialColumnFromDWAR("C:\\datasets\\mc2r_anta_a.dwar","Structure");
                Map<String,Double> randData = new HashMap<>();
                Random ri = new Random(123);
                for(String si : idcodes_a) {
                    randData.put(si,ri.nextDouble());
                }

                model.setCompounds(idcodes_a);
                model.setNumericData(randData);
            }


            runTestActions(model,2000);

            EchelonViewMain view = new EchelonViewMain(model);
            fa.getContentPane().setLayout(new BorderLayout());
            fa.getContentPane().add(view);
            fa.setVisible(true);
        });
    }

    public static void runTestActions(EchelonModel model, int waitMilliseconds) {
        Thread ti = new Thread() {
            @Override
            public void run() {
                try {
                    Thread.sleep(waitMilliseconds);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

                if(false) {
                    //String label_a = model.createNumericDecompositionProvider().getAllLabels().iterator().next();
                    ShowSinglePartAnalysisAction showAction = new ShowSinglePartAnalysisAction(model, null);
                    showAction.actionPerformed(new ActionEvent(null, -1, ""));
                }
            }
        };
        ti.start();
    }

    private void initMenu() {
        JMenu jma = new JMenu("Test");
        ShowSinglePartAnalysisAction showSinglePartAnalysisAction = new ShowSinglePartAnalysisAction(model,null);
        JMenuItem jmi = new JMenuItem(showSinglePartAnalysisAction);
        jma.add(jmi);
        this.menuBar.add(jma);
    }

    // proj cancer metabol dead..
    // dcMH@DdDfVulUZ`BH@MfrfxUPTqptRTlfgCQYpt^\MDWCQUpt]SMFzDu~
    // dcNH@IAIfU{EV``b@C[dpjxUgB@gzKdLsI\zZJUEahxsLesihiRVFcRptVRVSrptQVFbjpt]Qfc_L`Lu~

    // proj
    // dk^@@@RieUyngZ`hHh@Ma`jFUttGHnEKK]|rnEKK]|znEKK]|qnEKK]|ynEKK]|uRVCmpiY[of^J~Ph
    // dazD`La@BLddLruT@[HW_ifDSMEbU[E_HWfa|eRofs\b_u@Ys|
}
