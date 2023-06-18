package tech.molecules.leet.gui.chem.editor.sar;

import tech.molecules.leet.chem.sar.MultiFragment;
import tech.molecules.leet.chem.sar.SARElement;

import java.util.ArrayList;
import java.util.List;

public class SARDecompositionFragmentListModel {

    private List<MultiFragment> multiFragments = new ArrayList<>();
    private List<MultiFragmentListListener> multiFragmentListeners = new ArrayList<>();

    private SARElement editedElement = null;

    public void addMultiFragment(MultiFragment multiFragment) {
        multiFragments.add(multiFragment);
        fireMultiFragmentAdded(multiFragment);
    }

    public void removeMultiFragment(int index) {
        multiFragments.remove(index);
        fireMultiFragmentRemoved(index);
    }

    public List<MultiFragment> getMultiFragments() {
        return this.multiFragments;
    }

    public void addMultiFragmentListListener(MultiFragmentListListener listener) {
        multiFragmentListeners.add(listener);
    }

    public void removeMultiFragmentListener(MultiFragmentListListener listener) {
        multiFragmentListeners.remove(listener);
    }

    private void fireMultiFragmentAdded(MultiFragment multiFragment) {
        for (MultiFragmentListListener listener : multiFragmentListeners) {
            listener.onMultiFragmentAdded(multiFragment);
        }
    }

    private void fireMultiFragmentRemoved(int index) {
        for (MultiFragmentListListener listener : multiFragmentListeners) {
            listener.onMultiFragmentRemoved(index);
        }
    }

    public interface MultiFragmentListListener {
        void onMultiFragmentAdded(MultiFragment multiFragment);
        void onMultiFragmentRemoved(int index);

        void onSARElementSelected();
    }
}
