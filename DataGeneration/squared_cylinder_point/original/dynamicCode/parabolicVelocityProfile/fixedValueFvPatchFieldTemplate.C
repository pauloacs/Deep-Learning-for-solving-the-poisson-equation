/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) YEAR OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "fixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = 69ab4c6c17db3044042dbda9dbb3b665c1c1c61b
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void parabolicVelocityProfile_69ab4c6c17db3044042dbda9dbb3b665c1c1c61b(bool load)
    {
        if (load)
        {
            // code that can be explicitly executed after loading
        }
        else
        {
            // code that can be explicitly executed before unloading
        }
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    parabolicVelocityProfileFixedValueFvPatchVectorField
);


const char* const parabolicVelocityProfileFixedValueFvPatchVectorField::SHA1sum =
    "69ab4c6c17db3044042dbda9dbb3b665c1c1c61b";


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

parabolicVelocityProfileFixedValueFvPatchVectorField::
parabolicVelocityProfileFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF)
{
    if (false)
    {
        Info<<"construct parabolicVelocityProfile sha1: 69ab4c6c17db3044042dbda9dbb3b665c1c1c61b"
            " from patch/DimensionedField\n";
    }
}


parabolicVelocityProfileFixedValueFvPatchVectorField::
parabolicVelocityProfileFixedValueFvPatchVectorField
(
    const parabolicVelocityProfileFixedValueFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper)
{
    if (false)
    {
        Info<<"construct parabolicVelocityProfile sha1: 69ab4c6c17db3044042dbda9dbb3b665c1c1c61b"
            " from patch/DimensionedField/mapper\n";
    }
}


parabolicVelocityProfileFixedValueFvPatchVectorField::
parabolicVelocityProfileFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict)
{
    if (false)
    {
        Info<<"construct parabolicVelocityProfile sha1: 69ab4c6c17db3044042dbda9dbb3b665c1c1c61b"
            " from patch/dictionary\n";
    }
}


parabolicVelocityProfileFixedValueFvPatchVectorField::
parabolicVelocityProfileFixedValueFvPatchVectorField
(
    const parabolicVelocityProfileFixedValueFvPatchVectorField& ptf
)
:
    fixedValueFvPatchField<vector>(ptf)
{
    if (false)
    {
        Info<<"construct parabolicVelocityProfile sha1: 69ab4c6c17db3044042dbda9dbb3b665c1c1c61b"
            " as copy\n";
    }
}


parabolicVelocityProfileFixedValueFvPatchVectorField::
parabolicVelocityProfileFixedValueFvPatchVectorField
(
    const parabolicVelocityProfileFixedValueFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF)
{
    if (false)
    {
        Info<<"construct parabolicVelocityProfile sha1: 69ab4c6c17db3044042dbda9dbb3b665c1c1c61b "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

parabolicVelocityProfileFixedValueFvPatchVectorField::
~parabolicVelocityProfileFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy parabolicVelocityProfile sha1: 69ab4c6c17db3044042dbda9dbb3b665c1c1c61b\n";
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void parabolicVelocityProfileFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        Info<<"updateCoeffs parabolicVelocityProfile sha1: 69ab4c6c17db3044042dbda9dbb3b665c1c1c61b\n";
    }

//{{{ begin code
    #line 42 "/home/paulo/Desktop/Image-Based-CFD-Using-Deep-Learning/extrair_apenas_dados/original/0/U.boundaryField.inlet"
//const scalar t = this->db().time().value();
		//scalar step = min(1,0.2*t);

		scalar U_ave = 1, h=0.5;
		
		const fvPatch& boundaryPatch = patch();
		vectorField& field = *this;

		forAll(boundaryPatch, faceI) 
		{
			field[faceI] = vector( 1.5 * U_ave * ( 1 - Foam::pow(boundaryPatch.Cf()[faceI].y()/h ,2)), 0 , 0);
		}
//}}} end code

    this->fixedValueFvPatchField<vector>::updateCoeffs();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

