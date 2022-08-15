package tech.molecules.leet.clustering.action;

import com.actelion.research.chem.descriptor.*;
import com.actelion.research.chem.descriptor.pharmacophoretree.DescriptorHandlerPTree;
import tech.molecules.leet.clustering.ClusterAppModel;
import tech.molecules.leet.clustering.gui.JClusteringToolView;
import tech.molecules.leet.table.NStructureDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.PairwiseDistanceColumn;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class CreateClusteringToolAction extends AbstractAction {

    private NexusTableModel ntm;
    //private CreateClusteringToolActionConfig conf;

    private NStructureDataProvider dp;
    private JPanel plotPanel;
    public CreateClusteringToolAction(NexusTableModel ntm, NStructureDataProvider dpi, JPanel plotPanel) {
        this.ntm  = ntm;
        this.dp = dpi;
        this.plotPanel = plotPanel;
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        // init the clustering app
        ClusterAppModel capp = new ClusterAppModel(this.ntm,this.dp);

        // init the descriptor columns
        //this.ntm.addNexusColumn(conf.getDp(), PairwiseDistanceColumn.createFromDescriptor(conf.getDp(),new DescriptorHandlerFunctionalGroups(),ntm.getAllRows()));
        //this.ntm.addNexusColumn(conf.getDp(), PairwiseDistanceColumn.createFromDescriptor(conf.getDp(),new DescriptorHandlerSkeletonSpheres(),ntm.getAllRows()));
        //this.ntm.addNexusColumn(conf.getDp(), PairwiseDistanceColumn.createFromDescriptor(conf.getDp(),new DescriptorHandlerLongFFP512(),ntm.getAllRows()));
        //this.ntm.addNexusColumn(conf.getDp(), PairwiseDistanceColumn.createFromDescriptor(conf.getDp(),new DescriptorHandlerLongPFP512(),ntm.getAllRows()));

        // add classification column to ntm:
        this.ntm.addNexusColumn(this.dp,capp.getClassificationColumn());

        JPanel targetPanel = null;
        if(this.plotPanel==null) {
            JFrame fi = new JFrame();
            fi.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            fi.setSize(800,600);
            JPanel pmain = new JPanel();
            fi.getContentPane().setLayout(new BorderLayout());
            fi.getContentPane().add(pmain,BorderLayout.CENTER);
            targetPanel = pmain;
            fi.setVisible(true);
        }

        // init GUI of application
        JClusteringToolView view = new JClusteringToolView(capp);
        targetPanel.removeAll();
        targetPanel.setLayout(new BorderLayout());
        targetPanel.add(view,BorderLayout.CENTER);
    }

}
