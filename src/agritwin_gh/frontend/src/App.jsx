function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center">
      <div className="max-w-2xl w-full px-6 text-center">
        <h1 className="text-4xl font-bold text-green-700 mb-4">
          AgriTwin-GH
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Digital Twin System for Precision Greenhouse Horticulture
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
            <h2 className="font-semibold text-green-600 mb-1">Crop Health</h2>
            <p className="text-sm text-gray-500">Monitor disease risk and growth stages in real time.</p>
          </div>
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
            <h2 className="font-semibold text-green-600 mb-1">Digital Twin</h2>
            <p className="text-sm text-gray-500">Simulate and predict greenhouse conditions.</p>
          </div>
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
            <h2 className="font-semibold text-green-600 mb-1">MPC Control</h2>
            <p className="text-sm text-gray-500">Automated predictive control for resource optimisation.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
