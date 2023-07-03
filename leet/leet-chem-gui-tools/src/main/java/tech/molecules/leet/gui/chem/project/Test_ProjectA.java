package tech.molecules.leet.gui.chem.project;

import com.formdev.flatlaf.FlatLightLaf;
import tech.molecules.leet.chem.sar.SimpleSARDecompositionModel;
import tech.molecules.leet.chem.sar.SimpleSARProject;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.chem.sar.tree.SARProjectTreeNode;
import tech.molecules.leet.gui.chem.project.action.AddSeriesAction;
import tech.molecules.leet.gui.chem.project.action.EditMultiSynthonStructureAction;
import tech.molecules.leet.gui.chem.project.action.LoadProjectAction;
import tech.molecules.leet.gui.chem.project.action.SaveProjectAction;

import javax.swing.*;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeModel;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class Test_ProjectA {

    public static void main(String args[]) {
        FlatLightLaf.setup();
        try {
            UIManager.setLookAndFeel(new FlatLightLaf());
        } catch (Exception ex) {
            System.err.println("Failed to initialize LaF");
        }

        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        ProjectTreeController tc = new ProjectTreeController();
        //SimpleSARDecompositionModel model = new SimpleSARDecompositionModel(new ArrayList<>());
        SimpleSARProject project = new SimpleSARProject();
        SARProjectTreeNode projectRootNode = new SARProjectTreeNode(project,null);
        DefaultTreeModel tm = new DefaultTreeModel( projectRootNode);

        List<Class<?>> checkedClasses = new ArrayList<>();
        checkedClasses.add(SimpleSARSeries.class);
        checkedClasses.add(SimpleSARProject.class);
        List<Action> sarProjectActions = new ArrayList<>();
        sarProjectActions.add(new LoadProjectAction( (pi) -> tm.setRoot( new SARProjectTreeNode(pi,null) ) ));
        sarProjectActions.add(new SaveProjectAction());
        sarProjectActions.add(new AddSeriesAction( (xi) -> {project.getSeries().add(xi); tm.reload(projectRootNode);} ));
        List<Action> seriesActions = new ArrayList<>();
        seriesActions.add(new EditMultiSynthonStructureAction( null , () -> {
            JFrame f2 = new JFrame();
            f2.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            f2.getContentPane().setLayout(new BorderLayout());
            JPanel pi = new JPanel(); pi.setLayout(new BorderLayout());
            f2.getContentPane().add(pi,BorderLayout.CENTER);
            f2.setSize(600,600); f2.setVisible(true);
            return pi;
        } ));



        tc.setListOfCheckedClasses(checkedClasses);
        tc.setNodeTypeActions(SimpleSARProject.class,sarProjectActions);
        tc.setNodeTypeActions(SimpleSARSeries.class,seriesActions);

        SeriesManagerTreeView treeView = new SeriesManagerTreeView(tm,tc);

        //fi.getContentPane().add(editor, BorderLayout.CENTER);
        fi.getContentPane().add(treeView, BorderLayout.CENTER);
        fi.setSize(600,600);
        fi.setVisible(true);



    }

}
