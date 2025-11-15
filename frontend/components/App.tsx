import React, { useState } from 'react';
import Map from './Map';
import Sidebar from './Sidebar';
import { SelectedCluster } from '../types';

const App: React.FC = () => {
  const [selectedCluster, setSelectedCluster] = useState<SelectedCluster | null>(null);

  return (
    <>
      <Map setSelectedCluster={setSelectedCluster} />
      <Sidebar selectedCluster={selectedCluster} />
    </>
  );
};

export default App;
