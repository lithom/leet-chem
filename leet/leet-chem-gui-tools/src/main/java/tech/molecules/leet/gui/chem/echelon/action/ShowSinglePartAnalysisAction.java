package tech.molecules.leet.gui.chem.echelon.action;

import tech.molecules.leet.gui.chem.echelon.model.EchelonModel;
import tech.molecules.leet.gui.chem.echelon.model.PartAnalysisModel;
import tech.molecules.leet.gui.chem.echelon.view.PartDetailView;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Collections;
import java.util.List;

public class ShowSinglePartAnalysisAction extends AbstractAction {

    private EchelonModel model;
    private List<String> partLabels;

    public ShowSinglePartAnalysisAction(EchelonModel model, List<String> partLabels) {
        super("Show single part analysis");
        this.model = model;
        this.partLabels = partLabels;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        fi.setSize(600,600);

        if(partLabels == null) {
            String label_a = model.createNumericDecompositionProvider().getAllLabels().iterator().next();
            this.partLabels = Collections.singletonList(label_a);
        }

        PartAnalysisModel pam = model.getPartAnalysisModel();
        PartDetailView pdi = new PartDetailView(pam,partLabels);
        fi.getContentPane().setLayout(new BorderLayout());
        fi.getContentPane().add(pdi,BorderLayout.CENTER);
        fi.setVisible(true);
    }
}
